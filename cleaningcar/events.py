import base64
import hashlib
import json
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime
from math import hypot
from pathlib import Path

import cv2

from utils.upload_queue import SQLiteUploadQueue

from .constants import VEHICLE_LABEL_CN, WHEEL_CLASS_NAME_TO_CLEAN_VALUE, WHEEL_SIDE_TO_PHOTO_TYPE
from .log_throttle import WindowedLogThrottler
from .npu_monitor import format_npu_status, snapshot_npu_status
from .plate import is_valid_plate, normalize_plate_candidate_text, normalize_plate_text
from .resize_accel import resize_bgr
from .vision import box_iou, get_anchor_point

YELLOW_TRUCK_LABEL = 'yellow truck'
NON_YELLOW_OVERRIDE_LABELS = frozenset(label for label in VEHICLE_LABEL_CN.keys() if label != YELLOW_TRUCK_LABEL)
NON_YELLOW_OVERRIDE_SECONDS = 2.0
NON_YELLOW_HIGH_CONFIDENCE = 0.85
NON_YELLOW_HIGH_CONF_STREAK = 3
PLATE_RECOGNITION_ABNORMAL_REASONS = frozenset({"PLATE_MISSING", "PLATE_NOT_LOCKED", "PLATE_NOT_DETECTED"})

class EventUploader:
    def __init__(self, url=None, token=None, timeout=8.0, queue_path=None,
                 max_retries=10, base_delay=1.0, max_delay=60.0):
        self.url = (url or '').strip()
        self.token = token
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.base_delay = max(0.5, float(base_delay))
        self.max_delay = max(self.base_delay, float(max_delay))
        self.db = None
        self.thread = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        if self.url:
            db_path = Path(queue_path) if queue_path else (Path('./events') / 'upload_queue.db')
            self.db = SQLiteUploadQueue(db_path)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def enqueue(self, payload):
        if not self.db or payload is None:
            return
        self.db.enqueue(payload)
        self._wake.set()

    def _worker(self):
        while not self._stop.is_set():
            job = self.db.next_job() if self.db else None
            if not job:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            job_id, payload, retries = job
            try:
                self._send(payload)
            except Exception as exc:
                delay = min(self.base_delay * (2 ** retries), self.max_delay)
                if retries + 1 > self.max_retries:
                    print(f'[uploader] drop event after {retries} retries: {exc}')
                    if self.db:
                        self.db.mark_success(job_id)
                else:
                    print(f'[uploader] failed to send event (retry in {delay:.1f}s): {exc}')
                    if self.db:
                        self.db.mark_failure(job_id, retries + 1, delay)
                continue
            if self.db:
                self.db.mark_success(job_id)

    def _send(self, data):
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(self.url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        if self.token:
            req.add_header('Authorization', f'Bearer {self.token}')
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()

    def close(self):
        if not self.thread:
            if self.db:
                self.db.close()
            return
        self._stop.set()
        self._wake.set()
        self.thread.join(timeout=2.0)
        if self.db:
            pending = self.db.pending()
            if pending:
                print(f'[uploader] pending events retained in queue ({pending})')
            self.db.close()


class WheelPhotoUploader:
    def __init__(self, url=None, token=None, timeout=8.0, queue_path=None,
                 max_retries=10, base_delay=1.0, max_delay=60.0):
        self.url = (url or '').strip()
        self.token = token
        self.timeout = timeout
        self.max_retries = max(1, int(max_retries))
        self.base_delay = max(0.5, float(base_delay))
        self.max_delay = max(self.base_delay, float(max_delay))
        self.db = None
        self.thread = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        if self.url:
            db_path = Path(queue_path) if queue_path else (Path('./events') / 'wheel_photo_queue.db')
            self.db = SQLiteUploadQueue(db_path)
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def enqueue(self, payload):
        if not self.db or payload is None:
            return
        self.db.enqueue(payload)
        self._wake.set()

    def _worker(self):
        while not self._stop.is_set():
            job = self.db.next_job() if self.db else None
            if not job:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            job_id, payload, retries = job
            try:
                self._send(payload)
            except Exception as exc:
                delay = min(self.base_delay * (2 ** retries), self.max_delay)
                if retries + 1 > self.max_retries:
                    print(f'[wheel-photo-uploader] drop photo after {retries} retries: {exc}')
                    if self.db:
                        self.db.mark_success(job_id)
                else:
                    print(f'[wheel-photo-uploader] failed to send photo (retry in {delay:.1f}s): {exc}')
                    if self.db:
                        self.db.mark_failure(job_id, retries + 1, delay)
                continue
            if self.db:
                self.db.mark_success(job_id)

    def _send(self, data):
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(self.url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        if self.token:
            req.add_header('Authorization', f'Bearer {self.token}')
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            resp.read()

    def close(self):
        if not self.thread:
            if self.db:
                self.db.close()
            return
        self._stop.set()
        self._wake.set()
        self.thread.join(timeout=2.0)
        if self.db:
            pending = self.db.pending()
            if pending:
                print(f'[wheel-photo-uploader] pending photos retained in queue ({pending})')
            self.db.close()


class EventManager:
    def __init__(
        self,
        config,
        fps,
        frame_size,
        zone_manager,
        event_log_path=None,
        uploader=None,
        capture_mode='path',
        wheel_result_provider=None,
        wheel_photo_uploader=None,
        wheel_photo_base_dir=None,
        session_id='',
        wheel_photo_bucket_seconds=0.25,
        wheel_photo_min_score=0.3,
    ):
        self.config = config
        self.logic = config.get('logic', {})
        self.zone_mgr = zone_manager
        self.fps = fps
        self.frame_w, self.frame_h = frame_size
        self.camera_id = config.get('camera_id', 'CAM')
        self.capture_dir = Path(config.get('event_capture_dir', './captures'))
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir = Path(config.get('event_output_dir', './events'))
        self.enable_event_disk = bool(self.logic.get('enable_event_disk', False))
        if self.enable_event_disk or uploader:
            self.events_dir.mkdir(parents=True, exist_ok=True)
        self.tracks = {}
        self.timeout_frames = int(config.get('track_timeout_frames', 60))
        self.base_time = datetime.now()
        self.stationary_min_frames = int(config.get('stationary_min_frames', 0))
        self.stationary_speed_thresh = float(config.get('stationary_speed_thresh', 8.0))
        self.min_water_hit_frames_for_wash = int(self.logic.get('min_water_hit_frames_for_wash', 30))
        self.water_window_size = int(self.logic.get('water_window_size', 20))
        self.water_window_min_hits = int(self.logic.get('water_window_min_hits', 3))
        self.vehicle_shrink_ratio = float(config.get('vehicle_shrink_ratio', 0.35))
        self.vehicle_lock_min_votes = int(config.get('vehicle_lock_min_votes', 80))
        self.vehicle_lock_on_confirm = bool(config.get('vehicle_lock_on_confirm', True))
        shadow_cfg = dict(self.logic.get('shadow_plate_pool') or {})
        legacy_shadow_cfg = config.get('shadow_pool', {}) or {}
        for key, value in legacy_shadow_cfg.items():
            shadow_cfg.setdefault(key, value)
        self.shadow_max = int(shadow_cfg.get('max_candidates', 50))
        self.shadow_max_age = int(shadow_cfg.get('max_age_frames', 120))
        self.plate_lock_frames = max(1, int(self.logic.get('plate_lock_frames', 6)))
        self.plate_text_window_frames = max(
            self.plate_lock_frames,
            int(shadow_cfg.get('text_window_frames', min(self.shadow_max_age, 50))),
        )
        self.plate_text_margin_ratio = float(shadow_cfg.get('text_margin_ratio', 0.12))
        self.plate_text_switch_min_consecutive = max(
            self.plate_lock_frames,
            int(shadow_cfg.get('text_switch_min_consecutive', 6)),
        )
        self.plate_text_switch_gain_ratio = float(shadow_cfg.get('text_switch_gain_ratio', 1.2))
        self.plate_text_switch_margin_ratio = float(
            shadow_cfg.get('text_switch_margin_ratio', max(self.plate_text_margin_ratio + 0.05, 0.18))
        )
        self.plate_color_min_confidence = float(
            shadow_cfg.get('color_min_confidence', shadow_cfg.get('plate_color_min_confidence', 0.70))
        )
        self.plate_color_lock_frames = max(
            1,
            int(shadow_cfg.get('color_lock_frames', shadow_cfg.get('plate_color_lock_frames', 3))),
        )
        self.plate_color_window_frames = max(
            self.plate_color_lock_frames,
            int(shadow_cfg.get('color_window_frames', shadow_cfg.get('plate_color_window_frames', min(self.shadow_max_age, 50)))),
        )
        self.plate_color_switch_min_consecutive = max(
            self.plate_color_lock_frames,
            int(
                shadow_cfg.get(
                    'color_switch_min_consecutive',
                    shadow_cfg.get('plate_color_switch_min_consecutive', self.plate_color_lock_frames + 1),
                )
            ),
        )
        self.plate_color_switch_gain_ratio = float(
            shadow_cfg.get('color_switch_gain_ratio', shadow_cfg.get('plate_color_switch_gain_ratio', 1.2))
        )
        self.plate_color_switch_margin = float(
            shadow_cfg.get('color_switch_margin', shadow_cfg.get('plate_color_switch_margin', 0.5))
        )
        self.shadow_pool = {}
        self.event_log_path = Path(event_log_path) if event_log_path else None
        if self.event_log_path:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.event_log_path.exists():
                with self.event_log_path.open('w', encoding='utf-8') as f:
                    f.write('camera_id,track_id,type,capture_time,frame_idx,stationary_frames,'
                            'wash_duration,plate,vehicle,direction_code,direction_label,plate_is_guess,anchor_dwell_frames\n')
        self.uploader = uploader
        self.capture_mode = capture_mode
        self.capture_quality = int(config.get('event_capture_quality', 85) or 85)
        self.copy_track_last_frame = bool(self.logic.get('copy_track_last_frame', False))
        self.wheel_result_provider = wheel_result_provider
        self.wheel_photo_uploader = wheel_photo_uploader
        self.wheel_photo_base_dir = Path(wheel_photo_base_dir) if wheel_photo_base_dir else Path('/data/ftp')
        self.session_id = str(session_id or '').strip()
        self.wheel_photo_bucket_seconds = max(0.05, float(wheel_photo_bucket_seconds))
        self.wheel_photo_min_score = float(wheel_photo_min_score)
        wheel_cfg = config.get('wheel', {}) or {}
        try:
            self.wheel_bind_pre_start_seconds = max(0.0, float(wheel_cfg.get('bind_pre_start_seconds', 3.0)))
        except (TypeError, ValueError):
            self.wheel_bind_pre_start_seconds = 3.0
        try:
            self.wheel_bind_after_end_seconds = max(0.0, float(wheel_cfg.get('bind_after_end_seconds', 3.0)))
        except (TypeError, ValueError):
            self.wheel_bind_after_end_seconds = 3.0
        self.wheel_bind_require_active = bool(wheel_cfg.get('bind_require_active', True))
        try:
            self.wheel_bind_wait_seconds = max(0.0, float(wheel_cfg.get('bind_wait_seconds', 0.8)))
        except (TypeError, ValueError):
            self.wheel_bind_wait_seconds = 0.8
        try:
            self.wheel_bind_wait_poll_seconds = max(0.02, float(wheel_cfg.get('bind_wait_poll_seconds', 0.08)))
        except (TypeError, ValueError):
            self.wheel_bind_wait_poll_seconds = 0.08
        self.lane_name = config.get('lane_name', '冲洗')
        self.default_plate_color = config.get('default_plate_color', '')
        self.default_plate_color_conf = float(config.get('default_plate_color_conf', 0.0))
        self.default_cleanliness = int(config.get('default_cleanliness', 0))
        self.anchor_offset_ratio = float(self.logic.get('anchor_offset_ratio', 0.0))
        if self.anchor_offset_ratio < 0.0:
            self.anchor_offset_ratio = 0.0
        elif self.anchor_offset_ratio > 0.95:
            self.anchor_offset_ratio = 0.95
        self.zone_b_anchor_min_frames = int(self.logic.get('zone_b_anchor_min_frames', 0))
        if self.zone_b_anchor_min_frames < 0:
            self.zone_b_anchor_min_frames = 0
        quality_cfg = self.logic.get('event_track_quality')
        if isinstance(quality_cfg, dict):
            self.event_track_quality_cfg = quality_cfg
            self.event_track_quality_enabled = bool(quality_cfg.get('enabled', True))
        else:
            self.event_track_quality_cfg = {}
            self.event_track_quality_enabled = bool(quality_cfg) if quality_cfg is not None else False
        quality_cfg = self.event_track_quality_cfg
        self.quality_min_hits_type1 = max(1, int(quality_cfg.get('min_hits_type1', 12)))
        self.quality_fast_min_hits_type1 = max(1, int(quality_cfg.get('fast_vehicle_min_hits_type1', 6)))
        self.quality_min_avg_vehicle_conf = float(quality_cfg.get('min_avg_vehicle_conf', 0.62))
        self.quality_fast_min_avg_vehicle_conf = float(quality_cfg.get('fast_vehicle_min_avg_conf', 0.72))
        self.quality_plate_candidate_min_hits = max(1, int(quality_cfg.get('plate_candidate_min_hits', 2)))
        self.quality_plate_candidate_can_confirm_type1 = bool(
            quality_cfg.get('plate_candidate_can_confirm_type1', True)
        )
        self.quality_suppress_obvious_false_type5 = bool(quality_cfg.get('suppress_obvious_false_type5', True))
        self.quality_suspicious_cooldown_seconds = max(
            0.0,
            float(quality_cfg.get('suspicious_cooldown_seconds', 6.0)),
        )
        self.min_type5_zone_a_dwell = int(
            quality_cfg.get(
                'min_zone_a_dwell_type5',
                self.logic.get('min_zone_a_dwell_frames_for_type5', 0),
            )
        )
        if self.min_type5_zone_a_dwell < 0:
            self.min_type5_zone_a_dwell = 0
        self.min_type1_track_frames = int(self.logic.get('min_track_frames_for_type1', 5))
        if self.min_type1_track_frames < 0:
            self.min_type1_track_frames = 0
        self.wash_dwell_offset = 0.0
        self.min_type4_zone_b_dwell = int(self.logic.get('min_zone_b_dwell_frames_for_type4', 60))
        if self.min_type4_zone_b_dwell < 0:
            self.min_type4_zone_b_dwell = 0
        self.allowed_events = {1, 2, 3, 4, 5, 6}
        self.disable_plate_only_events = True
        self.single_lifecycle_events = True
        self.require_vehicle_type_for_events = bool(self.logic.get('require_vehicle_type_for_events', False))
        self.max_per_id_video_seconds = 600.0
        self.per_id_video_tail_seconds = 10.0
        self.pending_events = {}
        self.upload_buffer = {}
        self.upload_qualified = set()
        self.upload_log_full = None
        self.upload_log_sent = None
        if self.uploader:
            self.upload_log_full = self.events_dir / 'upload_log_full.csv'
            self.upload_log_sent = self.events_dir / 'upload_log_sent.csv'
            if not self.upload_log_full.exists():
                with self.upload_log_full.open('w', encoding='utf-8') as f:
                    f.write('capture_time,id,type,sent,payload\n')
            if not self.upload_log_sent.exists():
                with self.upload_log_sent.open('w', encoding='utf-8') as f:
                    f.write('capture_time,id,type,payload\n')
        self.frame_timing = {}
        self.log_throttler = WindowedLogThrottler()
        self.suspicious_type5_cooldown = deque(maxlen=64)
        self.latency_log_window_seconds = float(self.logic.get('latency_log_window_seconds', 10.0))
        self._capture_base64_cache = {}
        self._capture_metrics = {
            'saved': 0,
            'failed': 0,
            'encode_seconds': 0.0,
            'write_seconds': 0.0,
            'base64_seconds': 0.0,
            'read_seconds': 0.0,
        }

    def record_frame_timing(self, frame_idx, capture_ts, infer_ts):
        if capture_ts is None or infer_ts is None:
            return
        try:
            self.frame_timing[int(frame_idx)] = (float(capture_ts), float(infer_ts))
        except Exception:
            return

    def frame_timestamp(self, frame_idx):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _ingest_plate_candidate(
        self,
        track_id,
        track_state,
        text,
        frame_idx,
        conf=None,
        trusted=True,
        plate_color='',
        plate_color_conf=None,
        plate_type='',
        update_frame_idx=None,
    ):
        normalized_plate = normalize_plate_candidate_text(text)
        has_valid_plate_candidate = bool(normalized_plate and is_valid_plate(normalized_plate))
        try:
            candidate_frame = int(frame_idx)
        except (TypeError, ValueError):
            candidate_frame = 0
        if has_valid_plate_candidate:
            last_candidate_frame = int(track_state.get('plate_candidate_last_frame', -1) or -1)
            track_state['plate_candidate_hits'] = int(track_state.get('plate_candidate_hits', 0) or 0) + 1
            if candidate_frame >= last_candidate_frame:
                track_state['plate_text_latest'] = normalized_plate
                if last_candidate_frame >= 0 and candidate_frame - last_candidate_frame <= 1:
                    track_state['plate_candidate_consecutive'] = int(
                        track_state.get('plate_candidate_consecutive', 0) or 0
                    ) + 1
                else:
                    track_state['plate_candidate_consecutive'] = 1
                track_state['plate_candidate_latest'] = normalized_plate
                track_state['plate_candidate_last_frame'] = candidate_frame
            self._add_shadow_candidate(
                track_id,
                normalized_plate,
                conf,
                candidate_frame,
                trusted=trusted,
            )
            self._update_locked_plate_text(
                track_id,
                track_state,
                candidate_frame if update_frame_idx is None else update_frame_idx,
            )
        if plate_color:
            plate_color_latest = str(plate_color).strip()
            if plate_color_latest:
                track_state['plate_color_latest'] = plate_color_latest
                parsed_color_conf = 0.0
                if plate_color_conf is not None:
                    try:
                        parsed_color_conf = float(plate_color_conf)
                    except (TypeError, ValueError):
                        parsed_color_conf = 0.0
                track_state['plate_color_latest_conf'] = parsed_color_conf
                self._update_locked_plate_color(track_state, plate_color_latest, parsed_color_conf, candidate_frame)
        if plate_type:
            track_state['plate_type'] = str(plate_type)
        return has_valid_plate_candidate

    def _merge_plate_candidate_history(self, track_id, track_state, entries, frame_idx):
        merged = 0
        if not entries:
            return merged
        ordered_entries = sorted(entries, key=lambda item: int((item or {}).get('frame', frame_idx) or frame_idx))
        for entry in ordered_entries:
            if not isinstance(entry, dict):
                continue
            if self._ingest_plate_candidate(
                track_id,
                track_state,
                entry.get('text', ''),
                entry.get('frame', frame_idx),
                conf=entry.get('conf'),
                trusted=entry.get('trusted', True),
                plate_color=entry.get('plate_color', ''),
                plate_color_conf=entry.get('plate_color_conf'),
                plate_type=entry.get('plate_type', ''),
                update_frame_idx=frame_idx,
            ):
                merged += 1
        return merged

    def update_track(self, track_id, plate_box, vehicle_box, plate_text, frame_idx, frame,
                     water_boxes, water_active, is_plate, vehicle_label, vehicle_conf,
                     plate_conf, confirmed, cleaning_label='', anchor_point=None, plate_is_guess=False,
                     plate_color='', plate_color_conf=None, plate_type='', plate_candidate_history=None):
        if track_id <= 0:
            return
        st = self.tracks.setdefault(track_id, {
            'events': set(),
            'stationary_frames': 0,
            'speed_buf': deque(maxlen=6),
            'wash_duration': 0.0,
            'washing': False,
            'washing_candidate': False,
            'washing_confirmed': False,
            'water_detected': False,
            'water_hit_frames': 0,
            'water_window': deque(maxlen=20),
            'effective_wash_frames': 0,
            'last_frame_idx': frame_idx,
            'last_frame': None,
            'plate_text_latest': '',
            'plate_text_locked': '',
            'plate_text_locked_is_guess': False,
            'plate_text_switch_candidate': '',
            'plate_text_switch_streak': 0,
            'plate_text': '',
            'plate_is_guess': False,
            'plate_color_latest': '',
            'plate_color_latest_conf': 0.0,
            'plate_color_locked': '',
            'plate_color_locked_conf': 0.0,
            'plate_color_vote_history': deque(maxlen=160),
            'plate_color_votes': {},
            'plate_color_switch_candidate': '',
            'plate_color_switch_streak': 0,
            'plate_color': '',
            'plate_color_conf': 0.0,
            'plate_type': '',
            'plate_candidate_hits': 0,
            'plate_candidate_consecutive': 0,
            'plate_candidate_latest': '',
            'plate_candidate_last_frame': -1,
            'vehicle_cls': '',
            'last_plate_box': None,
            'last_vehicle_box': None,
            'confirmed': False,
            'plate_conf_history': [],
            'vehicle_conf_history': [],
            'vehicle_hit_frames': 0,
            'anchor_history': deque(maxlen=10),
            'center_jump_history': deque(maxlen=12),
            'wash_start_time': None,
            'wash_end_time': None,
            'lane': self.lane_name,
            'last_cleaning': '',
            'vehicle_cls_frozen': False,
            'class_counts': {},
            'vehicle_cls_locked': '',
            'vehicle_non_yellow_recent': deque(),
            'vehicle_high_conf_label': '',
            'vehicle_high_conf_count': 0,
            'last_vehicle_label': '',
            'zone_state': None,
            'last_anchor': None,
            'last_type3_frame': -1,
            'last_type4_frame': -1,
            'zone_b_enter_frame': -1,
            'zone_b_dwell_frames': 0,
            'zone_a_enter_frame': -1,
            'zone_a_dwell_frames': 0,
            'zone_a_seen': False,
            'zone_a_exited': False,
            'track_frame_count': 0,
            'abnormal_reasons': set(),
            'type2_qualified': False,
            'type2_qualified_frame': -1,
            'type5_suppressed_quality': False,
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
            'wheel_active': False,
            'wheel_activity_start_ts': None,
            'wheel_activity_last_ts': None,
            'wheel_activity_end_ts': None,
        })
        if self.single_lifecycle_events and st.get('closed'):
            st['last_frame_idx'] = frame_idx
            self._update_wheel_track_activity(track_id, st, frame_ts=time.time(), active=False)
            return
        st['track_frame_count'] = st.get('track_frame_count', 0) + 1
        st['last_frame_idx'] = frame_idx
        if frame is not None:
            st['last_frame'] = frame.copy() if self.copy_track_last_frame else frame
        freeze_label = st.get('vehicle_cls_frozen', False)
        if vehicle_box is not None and st.get('last_vehicle_box') is not None:
            prev = st['last_vehicle_box']
            prev_area = max(1.0, (prev[2] - prev[0]) * (prev[3] - prev[1]))
            new_area = max(1.0, (vehicle_box[2] - vehicle_box[0]) * (vehicle_box[3] - vehicle_box[1]))
            if new_area < prev_area * self.vehicle_shrink_ratio:
                freeze_label = True
        counts = st.get('class_counts') or {}
        override_label = self._get_non_yellow_vehicle_override(st, vehicle_label, vehicle_conf, frame_idx)
        if override_label:
            counts = self._apply_non_yellow_vehicle_override(st, counts, override_label)
            freeze_label = True
        elif vehicle_label and not freeze_label:
            counts[vehicle_label] = counts.get(vehicle_label, 0) + 1
            st['class_counts'] = counts
            locked, locked_count = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
            st['vehicle_cls_locked'] = locked
            st['vehicle_cls'] = locked
            if locked_count >= self.vehicle_lock_min_votes:
                st['vehicle_cls_frozen'] = True
        elif vehicle_label and st.get('vehicle_cls_locked') and vehicle_label == st['vehicle_cls_locked']:
            counts[vehicle_label] = counts.get(vehicle_label, 0) + 1
            st['class_counts'] = counts
        elif not st.get('vehicle_cls') and st.get('vehicle_cls_locked'):
            st['vehicle_cls'] = st['vehicle_cls_locked']
        if vehicle_label:
            st['last_vehicle_label'] = vehicle_label
            if not st.get('vehicle_cls'):
                st['vehicle_cls'] = vehicle_label
        if freeze_label and st.get('vehicle_cls_locked'):
            st['vehicle_cls_frozen'] = True
        prev_vehicle_box_for_motion = st.get('last_vehicle_box')
        if vehicle_box is not None:
            st['last_vehicle_box'] = vehicle_box
        if plate_box is not None:
            st['last_plate_box'] = plate_box
        self._merge_plate_candidate_history(track_id, st, plate_candidate_history, frame_idx)
        self._ingest_plate_candidate(
            track_id,
            st,
            plate_text,
            frame_idx,
            conf=plate_conf,
            trusted=not bool(plate_is_guess),
            plate_color=plate_color,
            plate_color_conf=plate_color_conf,
            plate_type=plate_type,
        )
        self._sync_plate_legacy_fields(st)
        if confirmed:
            st['confirmed'] = True
            if self.vehicle_lock_on_confirm and st.get('vehicle_cls_locked'):
                st['vehicle_cls_frozen'] = True
        if vehicle_box is not None or vehicle_conf is not None:
            st['vehicle_hit_frames'] = int(st.get('vehicle_hit_frames', 0) or 0) + 1
        if vehicle_conf is not None:
            history = st.get('vehicle_conf_history') or []
            history.append(float(vehicle_conf))
            if len(history) > 60:
                history.pop(0)
            st['vehicle_conf_history'] = history
        if plate_conf is not None:
            history = st.get('plate_conf_history') or []
            history.append(float(plate_conf))
            if len(history) > 60:
                history.pop(0)
            st['plate_conf_history'] = history
        if cleaning_label:
            st['last_cleaning'] = cleaning_label
        ref_box = vehicle_box or plate_box or st.get('last_vehicle_box') or st.get('last_plate_box')
        prev_box = prev_vehicle_box_for_motion
        dist = 0.0
        if ref_box is not None and prev_box is not None:
            cx = 0.5 * (ref_box[0] + ref_box[2])
            cy = 0.5 * (ref_box[1] + ref_box[3])
            px = 0.5 * (prev_box[0] + prev_box[2])
            py = 0.5 * (prev_box[1] + prev_box[3])
            dist = hypot(cx - px, cy - py)
            jump_history = st.get('center_jump_history')
            if not isinstance(jump_history, deque):
                jump_history = deque(maxlen=12)
                st['center_jump_history'] = jump_history
            frame_diag = max(1.0, hypot(self.frame_w, self.frame_h))
            jump_history.append(dist / frame_diag)
        st['speed_buf'].append(dist)
        avg_speed = sum(st['speed_buf']) / max(len(st['speed_buf']), 1)
        speed_thresh = self.stationary_speed_thresh
        if vehicle_box is None and plate_box is not None:
            speed_thresh *= 1.5
        if avg_speed <= speed_thresh:
            st['stationary_frames'] = min(st['stationary_frames'] + 1, 100000)
        else:
            st['stationary_frames'] = max(st['stationary_frames'] - 1, 0)

        if anchor_point is None:
            anchor_point = get_anchor_point(vehicle_box or plate_box or st.get('last_vehicle_box') or st.get('last_plate_box'),
                                            self.anchor_offset_ratio)
        zone_flags = {'enter_a': False, 'exit_a': False, 'enter_b': False, 'exit_b': False}
        zone_state = st.get('zone_state')
        if anchor_point:
            st['last_anchor'] = anchor_point
            anchor_history = st.get('anchor_history')
            if not isinstance(anchor_history, deque):
                anchor_history = deque(maxlen=10)
                st['anchor_history'] = anchor_history
            anchor_history.append((float(anchor_point[0]), float(anchor_point[1])))
        zone_state, zone_flags = self.zone_mgr.update_track(track_id, anchor_point, frame_idx)
        st['zone_state'] = zone_state

        timestamp = self.frame_timestamp(frame_idx)
        inside_a = bool(zone_state and zone_state.inside_a)
        inside_b = bool(zone_state and zone_state.inside_b)
        if inside_a or zone_flags.get('enter_a') or st.get('zone_a_dwell_frames', 0) > 0:
            st['zone_a_seen'] = True
        if zone_flags.get('exit_a'):
            st['zone_a_exited'] = True
        if zone_flags.get('enter_a'):
            st['zone_a_enter_frame'] = frame_idx
            st['zone_a_dwell_frames'] = 0
        enter_a_frame = st.get('zone_a_enter_frame', -1)
        zone_a_elapsed = 0
        if inside_a:
            if enter_a_frame < 0:
                st['zone_a_enter_frame'] = frame_idx
                enter_a_frame = frame_idx
            zone_a_elapsed = frame_idx - enter_a_frame
            st['zone_a_dwell_frames'] = zone_a_elapsed
        else:
            if enter_a_frame >= 0:
                zone_a_elapsed = frame_idx - enter_a_frame
                st['zone_a_dwell_frames'] = zone_a_elapsed
            st['zone_a_enter_frame'] = -1
        if zone_flags.get('enter_b'):
            st['zone_b_enter_frame'] = frame_idx
            st['zone_b_dwell_frames'] = 0
            st['water_detected'] = False
            st['wash_duration'] = 0.0
            st['water_hit_frames'] = 0
            win = st.get('water_window')
            if isinstance(win, deque):
                win.clear()
            else:
                win = deque(maxlen=self.water_window_size)
            st['water_window'] = win
            st['effective_wash_frames'] = 0
        enter_frame = st.get('zone_b_enter_frame', -1)
        anchor_elapsed = 0
        if inside_b:
            if enter_frame < 0:
                st['zone_b_enter_frame'] = frame_idx
                enter_frame = frame_idx
            anchor_elapsed = frame_idx - enter_frame
            st['zone_b_dwell_frames'] = anchor_elapsed
        else:
            if enter_frame >= 0:
                anchor_elapsed = frame_idx - enter_frame
                st['zone_b_dwell_frames'] = anchor_elapsed
            st['zone_b_enter_frame'] = -1
        meets_anchor_delay = (not inside_b) or (anchor_elapsed >= self.zone_b_anchor_min_frames)
        stable_inside_b = bool(inside_b and meets_anchor_delay)
        event_enabled = bool(
            st.get('zone_a_dwell_frames', 0) > 0
            or inside_a
            or zone_flags.get('enter_a')
            or zone_flags.get('exit_a')
        )
        st['washing_candidate'] = stable_inside_b
        type2_ready = bool(event_enabled and zone_flags.get('enter_b'))
        if type2_ready:
            st['type2_qualified'] = True
            if st.get('type2_qualified_frame', -1) < 0:
                st['type2_qualified_frame'] = frame_idx

        wheel_ref_ts = time.time()
        wheel_active_now = self._is_wheel_track_active(st)
        self._update_wheel_track_activity(track_id, st, frame_ts=wheel_ref_ts, active=wheel_active_now)
        if wheel_active_now or not self.wheel_bind_require_active:
            self._update_track_wheel_results(track_id, st, frame_ts=wheel_ref_ts)

        water_in_b = bool(st.get('type2_qualified') and water_boxes)
        if water_in_b:
            st['water_detected'] = True
            st['water_hit_frames'] = st.get('water_hit_frames', 0) + 1
            st['effective_wash_frames'] = st.get('effective_wash_frames', 0) + 1
        washing_now = bool(water_in_b)

        if self.disable_plate_only_events and is_plate and (vehicle_box is None and st.get('last_vehicle_box') is None):
            return
        can_type1 = self._can_emit_type1(st)
        if bool(zone_state and zone_state.inside_a) and 1 not in st['events'] and 1 in self.allowed_events and can_type1:
            self.emit_event(track_id, 1, frame_idx, frame, {'captureTime': timestamp}, st)
            st['events'].add(1)
        if type2_ready and 2 not in st['events'] and 2 in self.allowed_events:
            if 1 in self.allowed_events and 1 not in st['events']:
                backfill_type1 = self._can_emit_type1(st)
                if backfill_type1:
                    self.emit_event(track_id, 1, frame_idx, frame, {'captureTime': timestamp}, st)
                    st['events'].add(1)
            self.emit_event(track_id, 2, frame_idx, frame, {'captureTime': timestamp}, st)
            st['events'].add(2)
        if water_in_b and 3 not in st['events'] and 3 in self.allowed_events:
            st['washing_confirmed'] = True
            st['wash_start_time'] = timestamp
            self.emit_event(track_id, 3, frame_idx, frame, {
                'captureTime': timestamp,
                'washStartTime': timestamp,
            }, st)
            st['events'].add(3)
        st['wash_duration'] = self._compute_effective_wash_duration(st, frame_idx)
        can_type4 = False
        if zone_flags.get('exit_b'):
            if st.get('zone_b_dwell_frames', 0) >= self.min_type4_zone_b_dwell:
                can_type4 = True
        if event_enabled and st.get('type2_qualified') and can_type4 and 4 not in st['events'] and 4 in self.allowed_events:
            st['wash_end_time'] = st.get('wash_end_time') or timestamp
            duration_val = self._compute_effective_wash_duration(st, frame_idx)
            st['wash_duration'] = duration_val
            self.emit_event(track_id, 4, frame_idx, frame, {
                'captureTime': timestamp,
                'washDuration': round(duration_val, 2),
            }, st)
            st['events'].add(4)
        can_type5 = self._can_emit_type5(st)
        if zone_flags.get('exit_a') and 5 in self.allowed_events and 5 not in st['events'] and can_type5:
            if self._should_suppress_type5_quality(track_id, st, frame_idx):
                st['type5_suppressed_quality'] = True
                if self.single_lifecycle_events:
                    st['closed'] = True
            else:
                st['wash_end_time'] = st.get('wash_end_time') or timestamp
                duration_val = self._compute_effective_wash_duration(st, frame_idx)
                st['wash_duration'] = duration_val
                self._mark_type5_abnormal_reasons(st)
                self.emit_event(track_id, 5, frame_idx, frame, {
                    'captureTime': timestamp,
                    'washDuration': round(duration_val, 2),
                }, st)
                st['events'].add(5)
                if self.single_lifecycle_events:
                    st['closed'] = True

        st['washing'] = washing_now
        self._update_wheel_track_activity(track_id, st, frame_ts=time.time())
        if not st['washing']:
            st['washing_confirmed'] = False
        st['debug'] = {
            'state': 'washing' if st.get('washing') else 'idle',
            'stationary': st['stationary_frames'],
            'speed': round(avg_speed, 1),
            'wash_duration': round(st.get('wash_duration', 0.0), 1),
            'water': bool(washing_now),
            'plate': st.get('plate_text', ''),
            'zone_a': bool(zone_state and zone_state.inside_a),
            'zone_b': bool(zone_state and zone_state.inside_b),
            'zone_a_elapsed': zone_a_elapsed,
            'zone_b_elapsed': anchor_elapsed,
            'water_detected': bool(st.get('water_detected')),
        }

        elapsed_seconds = st.get('track_frame_count', 0) / max(self.fps, 1e-6)
        if elapsed_seconds >= self.max_per_id_video_seconds and not st.get('closed'):
            reasons = st.get('abnormal_reasons')
            if reasons is None:
                reasons = set()
                st['abnormal_reasons'] = reasons
            if 'OVER_10_MINUTES' not in reasons:
                reasons.add('OVER_10_MINUTES')
            if st.get('record_start_frame') is not None and st.get('record_stop_frame') is None:
                extra_frames = self._record_tail_frames()
                last_idx = st.get('last_frame_idx', frame_idx)
                stop_frame = last_idx + extra_frames
                prev_stop = st.get('record_stop_frame')
                if prev_stop is None or stop_frame > prev_stop:
                    st['record_stop_frame'] = stop_frame
            if self.uploader:
                track_key = f'{self.camera_id}_{track_id}'
                buffer = self.upload_buffer.pop(track_key, [])
                if buffer:
                    updated = []
                    for p in buffer:
                        payload = dict(p)
                        payload['isAbnormal'] = True
                        old_reason = str(payload.get('abnormalReason') or '').strip()
                        if old_reason:
                            parts = set(r for r in old_reason.split('|') if r)
                        else:
                            parts = set()
                        parts.add('OVER_10_MINUTES')
                        payload['abnormalReason'] = '|'.join(sorted(parts))
                        updated.append(payload)
                    self.upload_qualified.add(track_key)
                    for payload in updated:
                        sent_now = False
                        try:
                            self.uploader.enqueue(payload)
                            sent_now = True
                        except Exception:
                            sent_now = False
                        if self.upload_log_sent and sent_now:
                            try:
                                text = json.dumps(payload, ensure_ascii=False)
                                with self.upload_log_sent.open('a', encoding='utf-8') as f:
                                    f.write(f"{self.frame_timestamp(st.get('last_frame_idx', frame_idx))},{track_key},0,{text}\n")
                            except Exception:
                                pass
                        if self.upload_log_full:
                            try:
                                text = json.dumps(payload, ensure_ascii=False)
                                with self.upload_log_full.open('a', encoding='utf-8') as f:
                                    f.write(f"{self.frame_timestamp(st.get('last_frame_idx', frame_idx))},{track_key},0,1,{text}\n")
                            except Exception:
                                pass
                else:
                    self.upload_qualified.add(track_key)
            st['closed'] = True

    def flush_inactive(self, active_ids, frame_idx, on_track_timeout=None):
        active_ids = active_ids or set()
        to_remove = []
        for tid, st in self.tracks.items():
            if tid in active_ids:
                continue
            if frame_idx - st.get('last_frame_idx', frame_idx) >= self.timeout_frames:
                event_enabled = bool(st.get('zone_a_dwell_frames', 0) > 0)
                suppress_plate_only_events = bool(
                    self.disable_plate_only_events
                    and st.get('last_vehicle_box') is None
                    and int(st.get('vehicle_hit_frames', 0) or 0) <= 0
                )
                if (
                    4 not in st['events']
                    and 4 in self.allowed_events
                    and event_enabled
                    and not suppress_plate_only_events
                    and st.get('type2_qualified')
                    and st.get('zone_b_dwell_frames', 0) > 0
                ):
                    last_frame = st.get('last_frame_idx', frame_idx)
                    st['wash_end_time'] = st.get('wash_end_time') or self.frame_timestamp(last_frame)
                    duration_val = self._compute_effective_wash_duration(st, last_frame)
                    st['wash_duration'] = duration_val
                    self.emit_event(tid, 4, last_frame, st.get('last_frame'), {
                        'captureTime': self.frame_timestamp(last_frame),
                        'washDuration': round(duration_val, 2),
                    }, st)
                    st['events'].add(4)
                can_type5 = self._can_emit_type5(st)
                if (
                    5 not in st['events']
                    and 5 in self.allowed_events
                    and can_type5
                    and event_enabled
                    and not suppress_plate_only_events
                ):
                    last_frame_idx = st.get('last_frame_idx', frame_idx)
                    if self._should_suppress_type5_quality(tid, st, last_frame_idx):
                        st['type5_suppressed_quality'] = True
                    else:
                        timestamp = self.frame_timestamp(last_frame_idx)
                        st['wash_end_time'] = st.get('wash_end_time') or timestamp
                        duration_val = self._compute_effective_wash_duration(st, last_frame_idx)
                        st['wash_duration'] = duration_val
                        extras = {
                            'captureTime': timestamp,
                            'washDuration': round(duration_val, 2),
                        }
                        self._mark_type5_abnormal_reasons(st)
                        self.emit_event(tid, 5, last_frame_idx, st.get('last_frame'), extras, st)
                        st['events'].add(5)
                if st.get('record_start_frame') is not None and st.get('record_stop_frame') is None:
                    extra_frames = self._record_tail_frames()
                    last_idx = st.get('last_frame_idx', frame_idx)
                    st['record_stop_frame'] = last_idx + extra_frames
                if self.single_lifecycle_events and 5 in st['events']:
                    st['closed'] = True

                stop_f = st.get('record_stop_frame')
                tail_pending = stop_f is not None and frame_idx <= stop_f

                if on_track_timeout is not None and not tail_pending:
                    try:
                        on_track_timeout(tid, st)
                    except Exception:
                        pass

                self._update_wheel_track_activity(tid, st, frame_ts=time.time(), active=False)

                if not tail_pending:
                    to_remove.append(tid)
        for tid in to_remove:
            self.shadow_pool.pop(tid, None)
            self.zone_mgr.drop_track(tid)
            self.pending_events.pop(tid, None)
            key = f'{self.camera_id}_{tid}'
            self.upload_buffer.pop(key, None)
            self.tracks.pop(tid, None)

    def emit_event(self, track_id, event_type, frame_idx, frame, payload, track_state):
        if event_type not in self.allowed_events:
            return
        vehicle_type = payload.get('vehicleType') or self._resolve_vehicle_type(track_state)
        if self.require_vehicle_type_for_events and event_type in (3, 4):
            if not (vehicle_type and str(vehicle_type).strip()):
                pending = self.pending_events.setdefault(track_id, [])
                pending.append({
                    'event_type': event_type,
                    'frame_idx': frame_idx,
                    'frame': frame.copy() if frame is not None else None,
                    'payload': dict(payload),
                })
                return
        if self.require_vehicle_type_for_events and vehicle_type and str(vehicle_type).strip():
            self._flush_pending_events(track_id, vehicle_type, track_state)
        self._emit_event_core(track_id, event_type, frame_idx, frame, payload, track_state, vehicle_type)

    def _emit_event_core(self, track_id, event_type, frame_idx, frame, payload, track_state, vehicle_type):
        t0 = time.perf_counter()
        anchor_dwell = 0
        dbg = track_state.get('debug', {})
        if dbg:
            anchor_dwell = int(dbg.get('zone_b_elapsed', 0) or 0)
        capture_time = payload.get('captureTime') or self.frame_timestamp(frame_idx)
        try:
            track_state['last_event_capture_time'] = capture_time
        except Exception:
            pass
        if event_type == 1:
            prev_type1_time = track_state.get('type1_capture_time')
            if not prev_type1_time:
                track_state['type1_capture_time'] = capture_time
                try:
                    dt = datetime.strptime(capture_time, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.now()
                ts_str = dt.strftime("%Y%m%d%H%M")
                device_name = self.config.get('system', {}).get('device_id') or self.camera_id
                session_id = f"{device_name}-{ts_str}-{track_id}"
                track_state['session_id'] = session_id
        session_id = track_state.get('session_id')
        if not session_id:
            base_time = track_state.get('type1_capture_time') or capture_time
            try:
                dt = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = datetime.now()
            ts_str = dt.strftime("%Y%m%d%H%M")
            device_name = self.config.get('system', {}).get('device_id') or self.camera_id
            session_id = f"{device_name}-{ts_str}-{track_id}"
            track_state['session_id'] = session_id
        if event_type == 1:
            prev_start = track_state.get('record_start_frame')
            if prev_start is None or frame_idx < prev_start:
                track_state['record_start_frame'] = frame_idx
        if track_state.get('record_start_frame') is None:
            track_state['record_start_frame'] = frame_idx
        if event_type == 5:
            self._wait_for_wheel_results_for_type5(track_id, track_state)
            extra_frames = self._record_tail_frames()
            stop_frame = frame_idx + extra_frames
            prev_stop = track_state.get('record_stop_frame')
            if prev_stop is None or stop_frame > prev_stop:
                track_state['record_stop_frame'] = stop_frame
            self._enqueue_wheel_photos(track_state)
        try:
            track_state[f'last_event_t{event_type}_capture_time'] = capture_time
        except Exception:
            pass
        capture_path = self._save_event_capture(event_type, track_id, frame_idx, frame)
        t_capture = time.perf_counter()
        event = {
            'id': session_id,
            'trackId': track_id,
            'type': event_type,
            'captureTime': capture_time,
            'plateNumber': '',
            'vehicleType': vehicle_type,
            'washDuration': payload.get('washDuration', 0.0),
            'washStartTime': payload.get('washStartTime', ''),
            'captureImage': capture_path,
            'lane': self.lane_name,
            'anchorDwellFrames': anchor_dwell,
            'plateRecognitionAbnormal': False,
        }
        reasons = track_state.get('abnormal_reasons') if track_state else None
        if reasons:
            if isinstance(reasons, set):
                reasons_list = sorted(reasons)
            else:
                reasons_list = sorted(str(x) for x in reasons if x)
            if reasons_list:
                event['isAbnormal'] = True
                event['abnormalReason'] = '|'.join(reasons_list)
        plate_text, is_guess, plate_recognition_abnormal, plate_abnormal_reason = self._resolve_report_plate_fields(
            track_id,
            track_state,
            frame_idx,
        )
        event['plateNumber'] = plate_text
        event['plateIsGuess'] = bool(is_guess and event['plateNumber'])
        event['plateRecognitionAbnormal'] = bool(plate_recognition_abnormal)
        self._apply_plate_recognition_flags(event, plate_abnormal_reason)
        dir_code = 0
        dir_label = ''
        if event_type == 5:
            dir_code, dir_label = self._resolve_direction(track_state)
            event['direction'] = dir_code
            event['directionLabel'] = dir_label
        plate_color, plate_color_conf = self._infer_plate_color(track_state)
        event['plateColor'] = plate_color
        event['plateColorConfidence'] = plate_color_conf
        if event_type == 5:
            event['washEndTime'] = track_state.get('wash_end_time') or event['captureTime']
            event['videoEndTime'] = self.frame_timestamp(track_state.get('last_frame_idx', frame_idx))
            event['totalWashDuration'] = round(track_state.get('wash_duration', 0.0), 2)
            video_duration = 0.0
            type1_time = track_state.get('type1_capture_time')
            if type1_time:
                try:
                    dt1 = datetime.strptime(type1_time, "%Y-%m-%d %H:%M:%S")
                    dt5 = datetime.strptime(event['captureTime'], "%Y-%m-%d %H:%M:%S")
                    delta = (dt5 - dt1).total_seconds()
                    if delta > 0:
                        video_duration = round(delta, 2)
                except Exception:
                    video_duration = 0.0
            event['videoDuration'] = video_duration
            event['cleanliness'] = self.default_cleanliness
            self._attach_wheel_results(event, track_state=track_state, track_id=track_id)
        if track_state.get('wash_start_time') and not event.get('washStartTime'):
            event['washStartTime'] = track_state.get('wash_start_time')
        capture_ts_val = None
        infer_ts_val = None
        if hasattr(self, 'frame_timing'):
            t = self.frame_timing.get(int(frame_idx))
            if t:
                capture_ts_val, infer_ts_val = t
        if capture_ts_val is not None and infer_ts_val is not None:
            now_ts = time.time()
            decode_to_infer = max(0.0, infer_ts_val - capture_ts_val)
            infer_to_event = max(0.0, now_ts - infer_ts_val)
            total_latency = max(0.0, now_ts - capture_ts_val)
            try:
                cap_str = datetime.fromtimestamp(capture_ts_val).strftime("%H:%M:%S.%f")[:-3]
                infer_str = datetime.fromtimestamp(infer_ts_val).strftime("%H:%M:%S.%f")[:-3]
                event_str = datetime.fromtimestamp(now_ts).strftime("%H:%M:%S.%f")[:-3]
                for line in self.log_throttler.record(
                    key='latency.event',
                    message=(
                        f"[latency] 帧={frame_idx} 轨迹={track_id} 类型={event_type} "
                        f"捕获={cap_str} 推理完成={infer_str} 告警发送={event_str} "
                        f"解码→推理={decode_to_infer*1000:.1f}ms 推理→告警={infer_to_event*1000:.1f}ms "
                        f"总时延={total_latency*1000:.1f}ms"
                    ),
                    window_seconds=self.latency_log_window_seconds,
                ):
                    print(line)
            except Exception:
                pass
        if self.enable_event_disk:
            event_path = self.events_dir / f'{self.camera_id}_{track_id}_t{event_type}_{frame_idx}.json'
            try:
                with event_path.open('w', encoding='utf-8') as f:
                    json.dump(event, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        t_json = time.perf_counter()
        print(f"[EVENT] cam={self.camera_id} track={track_id} type={event_type} time={event['captureTime']}")
        if self.event_log_path:
            try:
                with self.event_log_path.open('a', encoding='utf-8') as f:
                    f.write(f"{self.camera_id},{track_id},{event_type},{event['captureTime']},{frame_idx},"
                            f"{track_state.get('stationary_frames',0)},{round(track_state.get('wash_duration',0.0),2)},"
                            f"{event['plateNumber']},{event['vehicleType']},{dir_code},{dir_label},"
                            f"{int(event['plateIsGuess'])},{anchor_dwell}\n")
            except Exception:
                pass
        t_csv = time.perf_counter()
        if self.uploader:
            api_payload = self._build_api_payload(event, track_state, frame_idx)
            if api_payload:
                track_key = event['id']
                sent_now = False
                if event_type == 1:
                    buffer = self.upload_buffer.setdefault(track_key, [])
                    buffer.append(api_payload)
                elif event_type == 2:
                    buffer = self.upload_buffer.pop(track_key, [])
                    buffer.append(api_payload)
                    self.upload_qualified.add(track_key)
                    for p in buffer:
                        self._refresh_plate_fields_for_payload(p, track_id, track_state, frame_idx)
                        try:
                            self.uploader.enqueue(p)
                        except Exception:
                            continue
                        sent_now = True
                        if self.upload_log_sent:
                            try:
                                text = json.dumps(p, ensure_ascii=False)
                                with self.upload_log_sent.open('a', encoding='utf-8') as f:
                                    f.write(f"{event['captureTime']},{track_key},{event_type},{text}\n")
                            except Exception:
                                pass
                elif track_key in self.upload_qualified:
                    try:
                        self.uploader.enqueue(api_payload)
                        sent_now = True
                        if self.upload_log_sent:
                            try:
                                text = json.dumps(api_payload, ensure_ascii=False)
                                with self.upload_log_sent.open('a', encoding='utf-8') as f:
                                    f.write(f"{event['captureTime']},{track_key},{event_type},{text}\n")
                            except Exception:
                                pass
                    except Exception:
                        sent_now = False
                else:
                    buffer = self.upload_buffer.setdefault(track_key, [])
                    buffer.append(api_payload)
                if self.upload_log_full:
                    try:
                        text = json.dumps(api_payload, ensure_ascii=False)
                        with self.upload_log_full.open('a', encoding='utf-8') as f:
                            f.write(f"{event['captureTime']},{track_key},{event_type},{int(sent_now)},{text}\n")
                    except Exception:
                        pass
        t_upload = time.perf_counter()
        if event_type == 3:
            track_state['last_type3_frame'] = frame_idx
        elif event_type == 4:
            track_state['last_type4_frame'] = frame_idx
        t_total = t_upload - t0
        print(
            f'[perf:event-detail] frame={frame_idx} track={track_id} type={event_type} '
            f'总耗时={t_total*1000:.1f}ms | '
            f'截图={ (t_capture-t0)*1000:.1f}ms '
            f'JSON写={ (t_json-t_capture)*1000:.1f}ms '
            f'CSV写={ (t_csv-t_json)*1000:.1f}ms '
            f'上传={ (t_upload-t_csv)*1000:.1f}ms'
        )

    def _flush_pending_events(self, track_id, vehicle_type, track_state):
        entries = self.pending_events.pop(track_id, None)
        if not entries:
            return
        for entry in entries:
            et = entry.get('event_type')
            fi = entry.get('frame_idx')
            fr = entry.get('frame')
            payload = dict(entry.get('payload') or {})
            if not payload.get('vehicleType'):
                payload['vehicleType'] = vehicle_type
            self._emit_event_core(track_id, et, fi, fr, payload, track_state, vehicle_type)

    def _add_shadow_candidate(self, track_id, text, conf, frame_idx, trusted=False):
        text = normalize_plate_candidate_text(text)
        if not text or not is_valid_plate(text):
            return
        pool = self.shadow_pool.setdefault(track_id, deque())
        pool.append({
            'text': text,
            'conf': float(conf) if conf is not None else 0.5,
            'frame': frame_idx,
            'trusted': bool(trusted),
        })
        while len(pool) > self.shadow_max:
            pool.popleft()
        while pool and frame_idx - pool[0]['frame'] > self.shadow_max_age:
            pool.popleft()

    @staticmethod
    def _plate_margin_ok(best_weight, second_weight, ratio):
        if best_weight <= 0:
            return False
        if second_weight <= 0:
            return True
        gap = best_weight - second_weight
        if gap <= 0:
            return False
        return (gap / max(best_weight, 1e-6)) >= float(ratio)

    def _rank_plate_shadow_candidates(self, track_id, frame_idx):
        pool = self.shadow_pool.get(track_id)
        if not pool:
            return []
        min_frame = frame_idx - self.plate_text_window_frames + 1
        stats = {}
        for entry in pool:
            text = entry.get('text', '')
            if not text:
                continue
            hit_frame = int(entry.get('frame', frame_idx))
            if hit_frame < min_frame:
                continue
            age = max(0, frame_idx - hit_frame)
            decay = max(0.35, 1.0 - age / max(self.plate_text_window_frames, 1))
            conf = float(entry.get('conf', 0.5) or 0.5)
            weight = max(conf, 0.05) * decay
            info = stats.setdefault(
                text,
                {
                    'hits': 0,
                    'weight': 0.0,
                    'trusted_hits': 0,
                    'trusted_weight': 0.0,
                    'latest_frame': -1,
                },
            )
            info['hits'] += 1
            info['weight'] += weight
            info['latest_frame'] = max(int(info.get('latest_frame', -1)), hit_frame)
            if bool(entry.get('trusted', False)):
                info['trusted_hits'] += 1
                info['trusted_weight'] += weight
        ranked = [
            (
                text,
                info['hits'],
                info['weight'],
                info['trusted_hits'],
                info['trusted_weight'],
                info['latest_frame'],
            )
            for text, info in stats.items()
            if info['hits'] > 0
        ]
        ranked.sort(key=lambda item: (item[3], item[4], item[1], item[2], item[5], item[0]), reverse=True)
        return ranked

    def _update_locked_plate_text(self, track_id, track_state, frame_idx):
        ranked = self._rank_plate_shadow_candidates(track_id, frame_idx)
        if not ranked:
            return
        best_text, best_hits, best_weight, best_trusted_hits, best_trusted_weight, _best_frame = ranked[0]
        locked_text = (track_state.get('plate_text_locked') or '').strip()
        if not locked_text:
            second_trusted_weight = 0.0
            for item in ranked[1:]:
                second_trusted_weight = max(second_trusted_weight, float(item[4]))
            margin_ok = self._plate_margin_ok(
                best_trusted_weight if best_trusted_weight > 0 else best_weight,
                second_trusted_weight,
                self.plate_text_margin_ratio,
            )
            if best_hits >= self.plate_lock_frames and best_trusted_hits > 0 and margin_ok:
                track_state['plate_text_locked'] = best_text
                track_state['plate_text_locked_is_guess'] = False
                track_state['plate_text_switch_candidate'] = ''
                track_state['plate_text_switch_streak'] = 0
            return
        if best_text == locked_text:
            track_state['plate_text_switch_candidate'] = ''
            track_state['plate_text_switch_streak'] = 0
            return
        if best_trusted_hits <= 0:
            track_state['plate_text_switch_candidate'] = ''
            track_state['plate_text_switch_streak'] = 0
            return
        locked_weight = 0.0
        for cand_text, _cand_hits, _cand_weight, _cand_trusted_hits, cand_trusted_weight, _cand_frame in ranked:
            if cand_text == locked_text:
                locked_weight = cand_trusted_weight
                break
        stronger_than_locked = best_trusted_weight >= max(
            locked_weight * self.plate_text_switch_gain_ratio,
            locked_weight + 0.05,
        )
        second_trusted_weight = 0.0
        for item in ranked[1:]:
            second_trusted_weight = max(second_trusted_weight, float(item[4]))
        margin_ok = self._plate_margin_ok(
            best_trusted_weight if best_trusted_weight > 0 else best_weight,
            second_trusted_weight,
            self.plate_text_switch_margin_ratio,
        )
        if stronger_than_locked and margin_ok:
            if track_state.get('plate_text_switch_candidate') == best_text:
                track_state['plate_text_switch_streak'] = int(track_state.get('plate_text_switch_streak', 0)) + 1
            else:
                track_state['plate_text_switch_candidate'] = best_text
                track_state['plate_text_switch_streak'] = 1
            if int(track_state.get('plate_text_switch_streak', 0)) >= self.plate_text_switch_min_consecutive:
                track_state['plate_text_locked'] = best_text
                track_state['plate_text_locked_is_guess'] = False
                track_state['plate_text_switch_candidate'] = ''
                track_state['plate_text_switch_streak'] = 0
            return
        track_state['plate_text_switch_candidate'] = ''
        track_state['plate_text_switch_streak'] = 0

    @staticmethod
    def _recent_color_streak(history, color, min_frame):
        streak = 0
        for entry in reversed(history):
            if int(entry.get('frame', -1)) < min_frame:
                break
            if entry.get('color') == color:
                streak += 1
                continue
            break
        return streak

    def _update_locked_plate_color(self, track_state, color, color_conf, frame_idx):
        color = (color or '').strip()
        if not color:
            return
        if color_conf < self.plate_color_min_confidence:
            return
        history = track_state.get('plate_color_vote_history')
        if not isinstance(history, deque):
            history = deque(maxlen=max(self.plate_color_window_frames * 4, 60))
            track_state['plate_color_vote_history'] = history
        history.append({'color': color, 'conf': float(color_conf), 'frame': int(frame_idx)})
        min_frame = frame_idx - self.plate_color_window_frames + 1
        while history and int(history[0].get('frame', -1)) < min_frame:
            history.popleft()

        votes = {}
        for entry in history:
            entry_color = (entry.get('color') or '').strip()
            if not entry_color:
                continue
            info = votes.setdefault(entry_color, {'hits': 0, 'sum_conf': 0.0})
            info['hits'] += 1
            info['sum_conf'] += float(entry.get('conf', 0.0) or 0.0)
        track_state['plate_color_votes'] = votes
        if not votes:
            return
        ranked = sorted(votes.items(), key=lambda kv: (kv[1]['sum_conf'], kv[1]['hits'], kv[0]), reverse=True)
        best_color, best_stats = ranked[0]
        best_hits = int(best_stats.get('hits', 0))
        best_sum = float(best_stats.get('sum_conf', 0.0))

        locked_color = (track_state.get('plate_color_locked') or '').strip()
        if not locked_color:
            if best_hits >= self.plate_color_lock_frames:
                track_state['plate_color_locked'] = best_color
                track_state['plate_color_locked_conf'] = max(best_sum / max(best_hits, 1), 0.0)
                track_state['plate_color_switch_candidate'] = ''
                track_state['plate_color_switch_streak'] = 0
            return

        if best_color == locked_color:
            track_state['plate_color_switch_candidate'] = ''
            track_state['plate_color_switch_streak'] = 0
            return

        locked_sum = float(votes.get(locked_color, {}).get('sum_conf', 0.0) or 0.0)
        stronger = best_sum >= max(
            locked_sum * self.plate_color_switch_gain_ratio,
            locked_sum + self.plate_color_switch_margin,
        )
        if not stronger or best_hits < (self.plate_color_lock_frames + 1):
            track_state['plate_color_switch_candidate'] = ''
            track_state['plate_color_switch_streak'] = 0
            return
        streak = self._recent_color_streak(history, best_color, min_frame)
        if track_state.get('plate_color_switch_candidate') == best_color:
            track_state['plate_color_switch_streak'] = max(int(track_state.get('plate_color_switch_streak', 0)), streak)
        else:
            track_state['plate_color_switch_candidate'] = best_color
            track_state['plate_color_switch_streak'] = streak
        if int(track_state.get('plate_color_switch_streak', 0)) >= self.plate_color_switch_min_consecutive:
            track_state['plate_color_locked'] = best_color
            track_state['plate_color_locked_conf'] = max(best_sum / max(best_hits, 1), 0.0)
            track_state['plate_color_switch_candidate'] = ''
            track_state['plate_color_switch_streak'] = 0

    def _sync_plate_legacy_fields(self, track_state):
        locked_text = (track_state.get('plate_text_locked') or '').strip()
        latest_text = (track_state.get('plate_text_latest') or '').strip()
        if locked_text:
            track_state['plate_text'] = locked_text
            track_state['plate_is_guess'] = bool(track_state.get('plate_text_locked_is_guess', False))
        elif latest_text:
            track_state['plate_text'] = latest_text
            track_state['plate_is_guess'] = True
        else:
            track_state['plate_text'] = ''
            track_state['plate_is_guess'] = False

        locked_color = (track_state.get('plate_color_locked') or '').strip()
        latest_color = (track_state.get('plate_color_latest') or '').strip()
        if locked_color:
            track_state['plate_color'] = locked_color
            track_state['plate_color_conf'] = float(track_state.get('plate_color_locked_conf', 0.0) or 0.0)
        elif latest_color:
            track_state['plate_color'] = latest_color
            track_state['plate_color_conf'] = float(track_state.get('plate_color_latest_conf', 0.0) or 0.0)
        else:
            track_state['plate_color'] = ''
            track_state['plate_color_conf'] = 0.0

    def _resolve_plate_with_shadow(self, track_id, track_state, frame_idx):
        plate_text, plate_is_guess, _is_abnormal, _reason = self._resolve_report_plate_fields(
            track_id,
            track_state,
            frame_idx,
        )
        return plate_text, plate_is_guess

    def _resolve_report_plate_fields(self, track_id, track_state, frame_idx):
        del frame_idx
        track_state = track_state or {}
        locked_text = normalize_plate_candidate_text((track_state.get('plate_text_locked') or '').strip())
        if locked_text and is_valid_plate(locked_text):
            return locked_text, False, False, ''
        latest_text = normalize_plate_candidate_text(
            (track_state.get('plate_text_latest') or track_state.get('plate_text') or '').strip()
        )
        pool = self.shadow_pool.get(track_id) or ()
        has_candidate = bool(latest_text and is_valid_plate(latest_text)) or any(
            is_valid_plate(normalize_plate_candidate_text(entry.get('text', ''))) for entry in pool
        )
        if has_candidate:
            return '', False, True, 'PLATE_NOT_LOCKED'
        return '', False, True, 'PLATE_NOT_DETECTED'

    @staticmethod
    def _reason_parts(reason_text):
        if not reason_text:
            return set()
        return {str(item).strip() for item in str(reason_text).split('|') if str(item).strip()}

    def _apply_plate_recognition_flags(self, payload, plate_abnormal_reason):
        reasons = self._reason_parts(payload.get('abnormalReason'))
        reasons -= PLATE_RECOGNITION_ABNORMAL_REASONS
        has_plate_abnormal = bool(plate_abnormal_reason)
        if has_plate_abnormal:
            reasons.add(str(plate_abnormal_reason).strip())
        payload['plateRecognitionAbnormal'] = has_plate_abnormal
        if reasons:
            payload['isAbnormal'] = True
            payload['abnormalReason'] = '|'.join(sorted(reasons))
        else:
            payload.pop('abnormalReason', None)
            payload['isAbnormal'] = False
        return payload

    def _refresh_plate_fields_for_payload(self, payload, track_id, track_state, frame_idx):
        plate_text, plate_is_guess, _is_abnormal, plate_abnormal_reason = self._resolve_report_plate_fields(
            track_id,
            track_state,
            frame_idx,
        )
        payload['plateNumber'] = plate_text
        payload['plateIsGuess'] = bool(plate_is_guess and plate_text)
        if 'plateConfidence' in payload and not plate_text:
            payload['plateConfidence'] = 0.0
        self._apply_plate_recognition_flags(payload, plate_abnormal_reason)
        return payload

    def _compute_effective_wash_duration(self, track_state, frame_idx):
        frames = track_state.get('effective_wash_frames', 0)
        if frames <= 0:
            return 0.0
        seconds = frames / max(self.fps, 1e-6)
        return max(0.0, seconds)

    def _record_tail_frames(self):
        return int(max(self.fps, 1.0) * self.per_id_video_tail_seconds)

    def _track_avg_vehicle_conf(self, track_state):
        return self._avg(track_state.get('vehicle_conf_history') or [])

    def _track_quality_hits(self, track_state):
        return max(
            int(track_state.get('vehicle_hit_frames', 0) or 0),
            int(track_state.get('track_frame_count', 0) or 0),
            int(track_state.get('zone_a_dwell_frames', 0) or 0),
        )

    def _has_valid_plate_candidate(self, track_state):
        if not track_state:
            return False
        locked_text = normalize_plate_text(track_state.get('plate_text_locked', ''))
        if locked_text and is_valid_plate(locked_text):
            return True
        latest_text = normalize_plate_candidate_text(track_state.get('plate_candidate_latest', ''))
        if latest_text and is_valid_plate(latest_text):
            if int(track_state.get('plate_candidate_hits', 0) or 0) >= self.quality_plate_candidate_min_hits:
                return True
            if int(track_state.get('plate_candidate_consecutive', 0) or 0) >= self.quality_plate_candidate_min_hits:
                return True
        return False

    def _direction_is_stable(self, track_state):
        history = track_state.get('anchor_history')
        if not isinstance(history, deque) or len(history) < 2:
            return False
        first = history[0]
        last = history[-1]
        dx = float(last[0]) - float(first[0])
        dy = float(last[1]) - float(first[1])
        net_dist = hypot(dx, dy)
        if net_dist < max(2.0, min(self.frame_w, self.frame_h) * 0.01):
            return False
        positive = 0
        negative = 0
        prev = history[0]
        for item in list(history)[1:]:
            sx = float(item[0]) - float(prev[0])
            sy = float(item[1]) - float(prev[1])
            step = sx * dx + sy * dy
            if step > 0:
                positive += 1
            elif step < 0:
                negative += 1
            prev = item
        total = positive + negative
        if total <= 0:
            return True
        return positive / max(total, 1) >= 0.7

    def _can_emit_type1(self, track_state):
        if not self.event_track_quality_enabled:
            if self.min_type1_track_frames <= 0:
                return True
            return int(track_state.get('zone_a_dwell_frames', 0) or 0) >= self.min_type1_track_frames

        hits = self._track_quality_hits(track_state)
        avg_conf = self._track_avg_vehicle_conf(track_state)
        if hits >= self.quality_min_hits_type1 and avg_conf >= self.quality_min_avg_vehicle_conf:
            return True
        if (
            hits >= self.quality_fast_min_hits_type1
            and avg_conf >= self.quality_fast_min_avg_vehicle_conf
            and self._direction_is_stable(track_state)
        ):
            return True
        if self.quality_plate_candidate_can_confirm_type1 and self._has_valid_plate_candidate(track_state):
            return True
        return False

    def _has_valid_zone_a_lifecycle_for_type5(self, track_state):
        if not self.event_track_quality_enabled:
            if self.min_type5_zone_a_dwell > 0:
                return track_state.get('zone_a_dwell_frames', 0) >= self.min_type5_zone_a_dwell
            return True
        if not track_state.get('zone_a_seen') and track_state.get('zone_a_dwell_frames', 0) <= 0:
            return False
        if track_state.get('zone_a_dwell_frames', 0) >= self.min_type5_zone_a_dwell:
            return True
        if 1 in track_state.get('events', set()):
            return True
        return self._can_emit_type1(track_state)

    def _max_center_jump_ratio(self, track_state):
        history = track_state.get('center_jump_history')
        if not isinstance(history, deque) or not history:
            return 0.0
        return max(float(v or 0.0) for v in history)

    def _is_obvious_false_type5(self, track_state):
        if self._has_valid_plate_candidate(track_state):
            return False
        hits = self._track_quality_hits(track_state)
        avg_conf = self._track_avg_vehicle_conf(track_state)
        dwell = int(track_state.get('zone_a_dwell_frames', 0) or 0)
        low_conf = avg_conf > 0.0 and avg_conf < self.quality_min_avg_vehicle_conf
        short_track = hits < self.quality_fast_min_hits_type1 or dwell < max(1, self.min_type5_zone_a_dwell)
        jumpy = self._max_center_jump_ratio(track_state) >= 0.18
        return bool(short_track and (low_conf or jumpy))

    def _suspicious_type5_area_key(self, track_state):
        box = track_state.get('last_vehicle_box') or track_state.get('last_plate_box')
        if not box or len(box) != 4:
            return ('unknown', 0, 0)
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        cell_w = max(1.0, self.frame_w / 4.0)
        cell_h = max(1.0, self.frame_h / 4.0)
        return (
            str(self.lane_name or ''),
            int(cx // cell_w),
            int(cy // cell_h),
        )

    def _should_suppress_type5_quality(self, track_id, track_state, frame_idx):
        del frame_idx
        if not self.event_track_quality_enabled or not self.quality_suppress_obvious_false_type5:
            return False
        if not self._is_obvious_false_type5(track_state):
            return False
        now = time.time()
        cutoff = now - self.quality_suspicious_cooldown_seconds
        while self.suspicious_type5_cooldown and self.suspicious_type5_cooldown[0].get('ts', 0.0) < cutoff:
            self.suspicious_type5_cooldown.popleft()
        area_key = self._suspicious_type5_area_key(track_state)
        duplicate = any(entry.get('area') == area_key for entry in self.suspicious_type5_cooldown)
        self.suspicious_type5_cooldown.append({'ts': now, 'area': area_key})
        suffix = ' duplicate' if duplicate else ''
        for line in self.log_throttler.record(
            key=f'event.type5.quality_suppressed.{area_key}',
            message=(
                f'[event-quality] suppress suspicious type5{suffix}: '
                f'track={track_id} hits={self._track_quality_hits(track_state)} '
                f'avg_conf={self._track_avg_vehicle_conf(track_state):.3f} '
                f'zone_a_dwell={track_state.get("zone_a_dwell_frames", 0)} '
                f'jump={self._max_center_jump_ratio(track_state):.3f}'
            ),
            window_seconds=max(self.quality_suspicious_cooldown_seconds, 1.0),
        ):
            print(line)
        return True

    def _can_emit_type5(self, track_state):
        if track_state.get('type5_suppressed_quality'):
            return False
        if not track_state.get('type2_qualified'):
            return False
        return self._has_valid_zone_a_lifecycle_for_type5(track_state)

    def _mark_type5_abnormal_reasons(self, track_state):
        reasons = track_state.get('abnormal_reasons')
        if reasons is None:
            reasons = set()
            track_state['abnormal_reasons'] = reasons
        if 2 not in track_state.get('events', set()):
            reasons.add('MISSING_TYPE2')
        if track_state.get('water_detected') and 3 not in track_state.get('events', set()):
            reasons.add('MISSING_TYPE3')
        if track_state.get('type2_qualified') and 4 not in track_state.get('events', set()):
            reasons.add('MISSING_TYPE4')
        return reasons

    def _avg(self, values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _non_yellow_override_window_frames(self):
        return max(1, int(round(max(self.fps, 1.0) * NON_YELLOW_OVERRIDE_SECONDS)))

    def _get_non_yellow_vehicle_override(self, track_state, vehicle_label, vehicle_conf, frame_idx):
        recent = track_state.get('vehicle_non_yellow_recent')
        if not isinstance(recent, deque):
            recent = deque()
            track_state['vehicle_non_yellow_recent'] = recent

        current_locked = track_state.get('vehicle_cls_locked') or track_state.get('vehicle_cls') or ''
        if current_locked != YELLOW_TRUCK_LABEL:
            recent.clear()
            track_state['vehicle_high_conf_label'] = ''
            track_state['vehicle_high_conf_count'] = 0
            return ''

        window_frames = self._non_yellow_override_window_frames()
        min_frame = frame_idx - window_frames + 1
        while recent and recent[0].get('frame', -1) < min_frame:
            recent.popleft()

        label = (vehicle_label or '').strip()
        try:
            conf_val = float(vehicle_conf if vehicle_conf is not None else 0.0)
        except (TypeError, ValueError):
            conf_val = 0.0

        if label in NON_YELLOW_OVERRIDE_LABELS:
            recent.append({'frame': frame_idx, 'label': label})
            if conf_val >= NON_YELLOW_HIGH_CONFIDENCE:
                if track_state.get('vehicle_high_conf_label') == label:
                    track_state['vehicle_high_conf_count'] = int(track_state.get('vehicle_high_conf_count', 0)) + 1
                else:
                    track_state['vehicle_high_conf_label'] = label
                    track_state['vehicle_high_conf_count'] = 1
            else:
                track_state['vehicle_high_conf_label'] = ''
                track_state['vehicle_high_conf_count'] = 0
        else:
            track_state['vehicle_high_conf_label'] = ''
            track_state['vehicle_high_conf_count'] = 0

        streak_label = track_state.get('vehicle_high_conf_label', '')
        streak_count = int(track_state.get('vehicle_high_conf_count', 0) or 0)
        if streak_label in NON_YELLOW_OVERRIDE_LABELS and streak_count >= NON_YELLOW_HIGH_CONF_STREAK:
            return streak_label

        label_counts = {}
        for entry in recent:
            entry_label = entry.get('label', '')
            if entry_label in NON_YELLOW_OVERRIDE_LABELS:
                label_counts[entry_label] = label_counts.get(entry_label, 0) + 1
        if not label_counts:
            return ''
        best_label, best_count = max(label_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count >= window_frames:
            return best_label
        return ''

    def _apply_non_yellow_vehicle_override(self, track_state, counts, override_label):
        counts = dict(counts or {})
        yellow_count = int(counts.get(YELLOW_TRUCK_LABEL, 0) or 0)
        counts[override_label] = max(int(counts.get(override_label, 0) or 0), yellow_count + 1, self.vehicle_lock_min_votes)
        track_state['class_counts'] = counts
        track_state['vehicle_cls_locked'] = override_label
        track_state['vehicle_cls'] = override_label
        track_state['vehicle_cls_frozen'] = True
        track_state['vehicle_high_conf_label'] = ''
        track_state['vehicle_high_conf_count'] = 0
        recent = track_state.get('vehicle_non_yellow_recent')
        if isinstance(recent, deque):
            recent.clear()
        return counts

    def _infer_plate_color(self, track_state):
        locked_color = (track_state.get('plate_color_locked') or '').strip()
        if locked_color:
            try:
                locked_conf = float(track_state.get('plate_color_locked_conf', 0.0) or 0.0)
            except (TypeError, ValueError):
                locked_conf = 0.0
            return locked_color, locked_conf
        latest_color = (track_state.get('plate_color_latest') or '').strip()
        if latest_color:
            try:
                latest_conf = float(track_state.get('plate_color_latest_conf', 0.0) or 0.0)
            except (TypeError, ValueError):
                latest_conf = 0.0
            return latest_color, latest_conf
        vehicle = (track_state.get('vehicle_cls') or '').lower()
        plate = (
            track_state.get('plate_text_locked')
            or track_state.get('plate_text_latest')
            or track_state.get('plate_text')
            or ''
        )
        length = len(plate)
        if vehicle == 'car':
            if length == 8:
                return '绿色', 0.9
            return '蓝色', 0.9
        if vehicle == 'blue truck':
            return '蓝色', 0.9
        if vehicle in ('yellow truck', 'dump truck'):
            return '黄色', 0.9
        if vehicle == 'wuxiao':
            return '蓝色', 0.9
        if self.default_plate_color:
            return self.default_plate_color, self.default_plate_color_conf
        return '', 0.0

    def _resolve_vehicle_type(self, track_state, fallback=''):
        if not track_state:
            return fallback or ''
        locked = track_state.get('vehicle_cls_locked', '')
        if locked:
            return locked
        current = track_state.get('vehicle_cls', '')
        if current:
            return current
        counts = track_state.get('class_counts') or {}
        if counts:
            locked = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if locked:
                return locked
        return fallback or track_state.get('last_vehicle_label', '') or ''

    def get_locked_vehicle(self, track_id):
        st = self.tracks.get(track_id)
        if not st:
            return ''
        return st.get('vehicle_cls_locked') or st.get('vehicle_cls', '')

    def get_locked_plate(self, track_id):
        st = self.tracks.get(track_id)
        if not st:
            return {
                'text': '',
                'plate_color': '',
                'plate_color_conf': 0.0,
                'plate_type': '',
            }
        text = normalize_plate_candidate_text(st.get('plate_text_locked', ''))
        if not is_valid_plate(text):
            text = ''
        color = (st.get('plate_color_locked') or '').strip()
        try:
            color_conf = float(st.get('plate_color_locked_conf', 0.0) or 0.0)
        except (TypeError, ValueError):
            color_conf = 0.0
        return {
            'text': text,
            'plate_color': color,
            'plate_color_conf': color_conf,
            'plate_type': st.get('plate_type', ''),
        }

    def get_track_debug(self, track_id):
        st = self.tracks.get(track_id)
        if not st:
            return None
        dbg = st.get('debug', {})
        return {
            'state': dbg.get('state', 'idle'),
            'stationary': dbg.get('stationary', 0),
            'speed': dbg.get('speed', 0.0),
            'water': dbg.get('water', False),
            'wash_duration': dbg.get('wash_duration', 0.0),
            'plate': dbg.get('plate', ''),
            'zone_a': dbg.get('zone_a', False),
            'zone_b': dbg.get('zone_b', False),
        }

    def _prepare_capture_image(self, capture_path):
        if not capture_path:
            return ''
        capture_file = Path(capture_path)
        if not capture_file.exists():
            return ''
        if self.capture_mode == 'base64':
            cached = self._capture_base64_cache.pop(str(capture_file), None)
            if cached:
                return cached
            try:
                t0 = time.perf_counter()
                with capture_file.open('rb') as f:
                    data = f.read()
                self._capture_metrics['read_seconds'] += max(0.0, time.perf_counter() - t0)
                t1 = time.perf_counter()
                encoded = base64.b64encode(data).decode('utf-8')
                self._capture_metrics['base64_seconds'] += max(0.0, time.perf_counter() - t1)
                return encoded
            except Exception:
                return ''
        return str(capture_file)

    def _log_capture_failure(self, event_type, track_id, frame_idx, capture_path, frame,
                             imwrite_ret=None, error=''):
        frame_shape = ''
        if frame is not None:
            try:
                h, w = frame.shape[:2]
                frame_shape = f'{w}x{h}'
            except Exception:
                frame_shape = 'unavailable'
        print(
            '[event-capture] failed '
            f'type={event_type} track={track_id} frame={frame_idx} '
            f'path={capture_path or ""} frame_none={frame is None} '
            f'frame_shape={frame_shape or "none"} imwrite={imwrite_ret} error={error or ""}'
        )

    def _save_event_capture(self, event_type, track_id, frame_idx, frame):
        _cap_t0 = time.perf_counter()
        if frame is None:
            self._log_capture_failure(event_type, track_id, frame_idx, '', frame, error='frame is None')
            self._capture_metrics['failed'] += 1
            return ''
        now = datetime.now()
        capture_dir = self.capture_dir / now.strftime('%Y%m%d') / now.strftime('%H')
        capture_dir.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime('%Y%m%d_%H%M%S')
        safe_camera_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(self.camera_id or 'CAM'))
        capture_file = capture_dir / f'{stamp}_{safe_camera_id}_{track_id}_t{event_type}_f{frame_idx}.jpg'
        capture_path = str(capture_file)
        try:
            h, w = frame.shape[:2]
            target_w, target_h = 1920, 1080
            _t_resize0 = time.perf_counter()
            if w != target_w or h != target_h:
                frame_to_save = resize_bgr(frame, (target_w, target_h))
            else:
                frame_to_save = frame
            _t_resize = time.perf_counter()
            quality = min(max(int(self.capture_quality), 1), 100)
            params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            encode_start = time.perf_counter()
            ok, encoded = cv2.imencode('.jpg', frame_to_save, params)
            _t_encode = time.perf_counter()
            self._capture_metrics['encode_seconds'] += max(0.0, _t_encode - encode_start)
            if not ok:
                try:
                    capture_file.unlink(missing_ok=True)
                except Exception:
                    pass
                self._capture_metrics['failed'] += 1
                self._log_capture_failure(event_type, track_id, frame_idx, capture_path, frame, imwrite_ret=False)
                return ''
            jpeg_bytes = encoded.tobytes()
            write_start = time.perf_counter()
            with capture_file.open('wb') as f:
                f.write(jpeg_bytes)
            _t_write = time.perf_counter()
            self._capture_metrics['write_seconds'] += max(0.0, _t_write - write_start)
            if self.capture_mode == 'base64' and self.uploader:
                b64_start = time.perf_counter()
                self._capture_base64_cache[capture_path] = base64.b64encode(jpeg_bytes).decode('utf-8')
                self._capture_metrics['base64_seconds'] += max(0.0, time.perf_counter() - b64_start)
            if not capture_file.exists() or capture_file.stat().st_size <= 0:
                try:
                    capture_file.unlink(missing_ok=True)
                except Exception:
                    pass
                self._capture_metrics['failed'] += 1
                self._log_capture_failure(
                    event_type,
                    track_id,
                    frame_idx,
                    capture_path,
                    frame,
                    imwrite_ret=True,
                    error='file missing after write',
                )
                return ''
            self._capture_metrics['saved'] += 1
            _t_total = _t_write - _cap_t0
            print(
                f'[perf:capture-detail] frame={frame_idx} track={track_id} type={event_type} '
                f'截图总={_t_total*1000:.1f}ms | '
                f'resize={(_t_resize-_t_resize0)*1000:.1f}ms '
                f'编码={(_t_encode-encode_start)*1000:.1f}ms '
                f'写盘={(_t_write-write_start)*1000:.1f}ms '
                f'size={len(jpeg_bytes)/1024:.1f}KB'
            )
            return capture_path
        except Exception as exc:
            try:
                capture_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._capture_metrics['failed'] += 1
            self._log_capture_failure(
                event_type,
                track_id,
                frame_idx,
                capture_path,
                frame,
                error=str(exc),
            )
            return ''

    def snapshot_metrics(self):
        metrics = dict(self._capture_metrics)
        metrics['active_tracks'] = len(self.tracks)
        metrics['pending_events'] = sum(len(v) for v in self.pending_events.values())
        metrics['upload_buffered'] = sum(len(v) for v in self.upload_buffer.values())
        metrics['capture_base64_cached'] = len(self._capture_base64_cache)
        return metrics

    def _build_api_payload(self, event, track_state, frame_idx):
        if not self.uploader:
            return None
        evt_type = event['type']
        if evt_type == 6:
            return {
                'id': event['id'],
                'type': evt_type,
                'lane': self.lane_name,
            }
        plate_conf = round(self._avg(track_state.get('plate_conf_history')), 3)
        vehicle_conf = round(self._avg(track_state.get('vehicle_conf_history')), 3)
        raw_vehicle_type = event.get('vehicleType') or self._resolve_vehicle_type(track_state)
        vehicle_type_cn = VEHICLE_LABEL_CN.get(raw_vehicle_type, raw_vehicle_type or '')
        capture_time = event['captureTime']
        capture_image = self._prepare_capture_image(event.get('captureImage'))
        lane = self.lane_name
        track_id = event.get('trackId')
        plate_number = event.get('plateNumber', '')
        plate_color = event.get('plateColor', self.default_plate_color)
        plate_color_conf = event.get('plateColorConfidence', self.default_plate_color_conf)
        plate_is_guess = event.get('plateIsGuess', False)
        plate_recognition_abnormal = bool(event.get('plateRecognitionAbnormal', False))
        reasons = track_state.get('abnormal_reasons') if track_state else None
        wash_start_time = track_state.get('wash_start_time') if track_state else None
        dir_code = 0
        dir_label = ''
        wash_end_time = None
        video_end_time = None
        total_wash_duration = None
        cleanliness = None
        video_duration = None
        if evt_type == 5:
            dir_code, dir_label = self._resolve_direction(track_state)
            wash_end_time = track_state.get('wash_end_time') or capture_time
            video_end_time = self.frame_timestamp(track_state.get('last_frame_idx', frame_idx))
            total_wash_duration = round(track_state.get('wash_duration', 0.0), 2)
            cleanliness = self.default_cleanliness
            video_duration = event.get('videoDuration')
        payload = {}
        payload['id'] = event['id']
        payload['type'] = evt_type
        payload['captureTime'] = capture_time
        payload['captureImage'] = capture_image
        if evt_type == 1:
            payload['lane'] = lane
            payload['plateNumber'] = plate_number
            payload['plateConfidence'] = plate_conf
            payload['plateColor'] = plate_color
            payload['plateColorConfidence'] = plate_color_conf
            payload['vehicleType'] = vehicle_type_cn
            payload['vehicleTypeConfidence'] = vehicle_conf
            payload['plateIsGuess'] = plate_is_guess
            payload['plateRecognitionAbnormal'] = plate_recognition_abnormal
            if wash_start_time:
                payload['washStartTime'] = wash_start_time
        elif evt_type in (2, 3, 4):
            payload['lane'] = lane
            payload['plateNumber'] = plate_number
            payload['plateConfidence'] = plate_conf
            payload['plateColor'] = plate_color
            payload['plateColorConfidence'] = plate_color_conf
            payload['vehicleType'] = vehicle_type_cn
            payload['vehicleTypeConfidence'] = vehicle_conf
            payload['plateIsGuess'] = plate_is_guess
            payload['plateRecognitionAbnormal'] = plate_recognition_abnormal
            if wash_start_time:
                payload['washStartTime'] = wash_start_time
        elif evt_type == 5:
            payload['washEndTime'] = wash_end_time
            payload['videoEndTime'] = video_end_time
            payload['totalWashDuration'] = total_wash_duration
            payload['cleanliness'] = cleanliness
            payload['videoDuration'] = video_duration
            payload['plateNumber'] = plate_number
            payload['plateConfidence'] = plate_conf
            payload['plateColor'] = plate_color
            payload['plateColorConfidence'] = plate_color_conf
            payload['vehicleType'] = vehicle_type_cn
            payload['vehicleTypeConfidence'] = vehicle_conf
            payload['lane'] = lane
            payload['plateIsGuess'] = plate_is_guess
            payload['plateRecognitionAbnormal'] = plate_recognition_abnormal
            if wash_start_time:
                payload['washStartTime'] = wash_start_time
            payload['direction'] = dir_code
            payload['directionLabel'] = dir_label
            self._attach_wheel_results(payload, track_state=track_state, track_id=track_id)
        else:
            payload['lane'] = lane
            payload['plateNumber'] = plate_number
            payload['plateConfidence'] = plate_conf
            payload['plateColor'] = plate_color
            payload['plateColorConfidence'] = plate_color_conf
            payload['vehicleType'] = vehicle_type_cn
            payload['vehicleTypeConfidence'] = vehicle_conf
            payload['plateIsGuess'] = plate_is_guess
            payload['plateRecognitionAbnormal'] = plate_recognition_abnormal
            if wash_start_time:
                payload['washStartTime'] = wash_start_time
        if not plate_number:
            payload['plateConfidence'] = 0.0
        if reasons:
            if isinstance(reasons, set):
                reasons_list = sorted(reasons)
            else:
                reasons_list = sorted(str(x) for x in reasons if x)
            if reasons_list:
                payload['isAbnormal'] = True
                payload['abnormalReason'] = '|'.join(reasons_list)
        _plate_text, _plate_is_guess, _plate_is_abnormal, plate_abnormal_reason = self._resolve_report_plate_fields(
            track_id,
            track_state,
            frame_idx,
        )
        self._apply_plate_recognition_flags(payload, plate_abnormal_reason)
        return payload

    def _attach_wheel_results(self, payload, track_state=None, track_id=None):
        if not isinstance(payload, dict):
            return payload
        wheel_results = self._build_wheel_results_payload(track_state=track_state, track_id=track_id)
        if wheel_results:
            payload['wheelResults'] = wheel_results
        return payload

    def _absolute_wheel_photo_url(self, photo_url):
        photo_url = str(photo_url or '').strip()
        if not photo_url:
            return ''
        photo_path = Path(photo_url)
        if photo_path.is_absolute():
            return photo_path.as_posix()
        return (self.wheel_photo_base_dir / photo_path).resolve().as_posix()

    def _find_existing_wheel_photo_url(self, track_state, side, entry):
        if not isinstance(track_state, dict) or not isinstance(entry, dict):
            return ''
        history = track_state.get('wheel_photo_history')
        if not isinstance(history, dict):
            return ''
        side_history = history.get(side)
        if not isinstance(side_history, dict):
            return ''
        entry_id = int(entry.get('entryId', 0) or 0)
        image_bytes = entry.get('imageJpegBytes', b'') or b''
        image_hash = hashlib.sha1(image_bytes).hexdigest() if image_bytes else ''
        for bucket in side_history.values():
            if not isinstance(bucket, dict):
                continue
            rep = bucket.get('representative')
            if not isinstance(rep, dict):
                continue
            photo_url = str(rep.get('photoUrl') or '').strip()
            if not photo_url:
                continue
            if entry_id > 0 and int(rep.get('entryId', 0) or 0) == entry_id:
                return self._absolute_wheel_photo_url(photo_url)
            if image_hash and str(rep.get('imageHash') or '') == image_hash:
                return self._absolute_wheel_photo_url(photo_url)
        return ''

    def _ensure_locked_wheel_photo_url(self, side, entry, track_state=None, track_id=None):
        if not isinstance(entry, dict):
            return ''
        photo_url = str(entry.get('photoUrl') or '').strip()
        if photo_url:
            photo_url = self._absolute_wheel_photo_url(photo_url)
            entry['photoUrl'] = photo_url
            return photo_url
        photo_url = self._find_existing_wheel_photo_url(track_state, side, entry)
        if photo_url:
            entry['photoUrl'] = photo_url
            return photo_url
        image_bytes = entry.get('imageJpegBytes', b'') or b''
        if not image_bytes:
            return ''
        if isinstance(track_state, dict):
            seq_map = track_state.setdefault('wheel_photo_seq', {'left': 0, 'right': 0})
        else:
            seq_map = {'left': 0, 'right': 0}
        seq = int(seq_map.get(side, 0) or 0) + 1
        seq_map[side] = seq
        photo_url = self._save_wheel_photo(
            side=side,
            track_id=track_id or 0,
            seq=seq,
            image_bytes=image_bytes,
            capture_ts=entry.get('capture_ts'),
            class_name=entry.get('className', ''),
        )
        if photo_url:
            entry['photoUrl'] = photo_url
        return photo_url or ''

    def _serialize_locked_wheel_entry(self, side, entry, track_state=None, track_id=None):
        side = str(side or '').strip().lower()
        if side not in ('left', 'right') or not isinstance(entry, dict):
            return None
        capture_time = str(entry.get('captureTime') or '').strip()
        class_name = str(entry.get('className') or '').strip()
        photo_url = self._ensure_locked_wheel_photo_url(
            side=side,
            entry=entry,
            track_state=track_state,
            track_id=track_id,
        )
        if not capture_time or not class_name or not photo_url:
            return None
        return {
            'side': side,
            'captureTime': capture_time,
            'photoUrl': photo_url,
            'className': class_name,
        }

    def _build_locked_wheel_results_payload(self, track_state, track_id=None):
        if not isinstance(track_state, dict):
            return []
        locked = track_state.get('wheel_results_locked')
        if not isinstance(locked, dict):
            return []
        results = []
        for side in ('left', 'right'):
            item = self._serialize_locked_wheel_entry(
                side,
                locked.get(side),
                track_state=track_state,
                track_id=track_id,
            )
            if item:
                results.append(item)
        return results

    def _build_wheel_results_payload(self, track_state=None, track_id=None):
        return self._build_locked_wheel_results_payload(track_state, track_id=track_id)

    @staticmethod
    def _is_wheel_track_active(track_state):
        if not isinstance(track_state, dict):
            return False
        if track_state.get('closed'):
            return False
        if 5 in (track_state.get('events') or ()):
            return False
        if int(track_state.get('zone_a_enter_frame', -1) or -1) >= 0:
            return True
        return bool(track_state.get('zone_a_dwell_frames', 0) > 0)

    def _update_wheel_track_activity(self, track_id, track_state, frame_ts=None, active=None):
        provider = getattr(self, 'wheel_result_provider', None)
        ref_ts = time.time() if frame_ts is None else float(frame_ts)
        if active is None:
            active = self._is_wheel_track_active(track_state)
        active = bool(active)
        if isinstance(track_state, dict):
            was_active = bool(track_state.get('wheel_active', False))
            if active:
                if not was_active or not track_state.get('wheel_activity_start_ts'):
                    track_state['wheel_activity_start_ts'] = ref_ts
                track_state['wheel_activity_last_ts'] = ref_ts
                track_state['wheel_activity_end_ts'] = None
            else:
                if was_active:
                    track_state['wheel_activity_end_ts'] = ref_ts
                track_state['wheel_activity_last_ts'] = ref_ts
            track_state['wheel_active'] = active
        if provider is None:
            return
        updater = getattr(provider, 'update_track_activity', None)
        if not callable(updater):
            return
        try:
            updater(track_id, track_state=track_state, frame_ts=ref_ts, active=active)
        except TypeError:
            try:
                updater(track_id, track_state, ref_ts, active)
            except Exception:
                return
        except Exception:
            return

    def _save_wheel_photo(self, side, track_id, seq, image_bytes, capture_ts, class_name):
        if not image_bytes:
            return None
        try:
            capture_dt = datetime.fromtimestamp(float(capture_ts))
        except (TypeError, ValueError, OSError):
            capture_dt = datetime.now()
        session = self.session_id or capture_dt.strftime('%H%M%S')
        date_dir = capture_dt.strftime('%Y%m%d')
        hour_dir = capture_dt.strftime('%H')
        rel_path = Path('box') / date_dir / hour_dir / f'{session}_{int(track_id)}_{side}_{int(seq)}.jpg'
        abs_path = self.wheel_photo_base_dir / rel_path
        try:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            with abs_path.open('wb') as f:
                f.write(image_bytes)
        except Exception as exc:
            print(f'[wheel-photo] failed to save {abs_path}: {exc}')
            return None
        return abs_path.resolve().as_posix()

    def _wheel_candidate_allowed_for_track(self, track_state, candidate, ref_ts):
        try:
            capture_ts = float(candidate.get('capture_ts', ref_ts) or ref_ts)
        except (TypeError, ValueError):
            capture_ts = ref_ts
        start_ts = track_state.get('wheel_activity_start_ts') if isinstance(track_state, dict) else None
        if start_ts is not None:
            try:
                start_ts = float(start_ts)
            except (TypeError, ValueError):
                start_ts = None
        if start_ts is not None and capture_ts < (start_ts - self.wheel_bind_pre_start_seconds):
            return False, 'before_track_start'
        end_ts = track_state.get('wheel_activity_end_ts') if isinstance(track_state, dict) else None
        if end_ts is not None:
            try:
                end_ts = float(end_ts)
            except (TypeError, ValueError):
                end_ts = None
        if end_ts is not None and capture_ts > (end_ts + self.wheel_bind_after_end_seconds):
            return False, 'after_track_end'
        if capture_ts > (float(ref_ts) + self.wheel_bind_after_end_seconds):
            return False, 'future_capture'
        return True, ''

    def _log_wheel_bind_decision(self, track_id, action, candidate, ref_ts, reason=''):
        if not isinstance(candidate, dict):
            return
        side = str(candidate.get('side') or '').strip().lower()
        entry_id = int(candidate.get('entryId', 0) or 0)
        try:
            capture_ts = float(candidate.get('capture_ts', ref_ts) or ref_ts)
        except (TypeError, ValueError):
            capture_ts = float(ref_ts)
        age = float(ref_ts) - capture_ts
        track_state = self.tracks.get(track_id) if hasattr(self, 'tracks') else None
        start_ts = None
        if isinstance(track_state, dict) and track_state.get('wheel_activity_start_ts') is not None:
            try:
                start_ts = float(track_state.get('wheel_activity_start_ts'))
            except (TypeError, ValueError):
                start_ts = None
        since_start = capture_ts - start_ts if start_ts is not None else 0.0
        reason_part = f' reason={reason}' if reason else ''
        msg = (
            f"[wheel-bind] track={track_id} side={side} action={action}{reason_part} "
            f"entry={entry_id} capture={candidate.get('captureTime', '')} "
            f"age={age:.2f}s since_start={since_start:.2f}s "
            f"class={candidate.get('className', '')} score={float(candidate.get('score', 0.0) or 0.0):.3f}"
        )
        key_reason = reason or action
        for line in self.log_throttler.record(
            key=f'wheel_bind.{action}.{side}.{key_reason}.{entry_id}',
            message=msg,
            now=time.time(),
            window_seconds=5.0,
        ):
            print(line)

    def _expected_wheel_side_count(self):
        provider = getattr(self, 'wheel_result_provider', None)
        active_sides = getattr(provider, 'active_sides', None)
        if isinstance(active_sides, (list, tuple, set)):
            sides = {str(side).strip().lower() for side in active_sides}
            return max(1, min(2, len(sides & {'left', 'right'})))
        return 2

    @staticmethod
    def _locked_wheel_side_count(track_state):
        locked = track_state.get('wheel_results_locked') if isinstance(track_state, dict) else None
        if not isinstance(locked, dict):
            return 0
        return sum(1 for side in ('left', 'right') if isinstance(locked.get(side), dict))

    @staticmethod
    def _fmt_age(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return '-'
        if value < 0.0:
            return '-'
        return f'{value:.1f}s'

    def _format_wheel_runtime_stats(self):
        provider = getattr(self, 'wheel_result_provider', None)
        snapshot = getattr(provider, 'snapshot_stats', None)
        if not callable(snapshot):
            return 'unavailable'
        try:
            stats = snapshot()
        except Exception as exc:
            return f'error:{exc}'
        service = stats.get('_service') if isinstance(stats, dict) else {}
        cache = stats.get('_cache') if isinstance(stats, dict) else {}
        chunks = [
            (
                'service('
                f"active={int(bool((service or {}).get('inference_active', False)))},"
                f"boost={int(bool((service or {}).get('boost_active', False)))},"
                f"tracks={(service or {}).get('active_track_count', 0)}"
                ')'
            )
        ]
        for side in ('left', 'right'):
            side_stats = stats.get(side) if isinstance(stats, dict) else {}
            cache_stats = cache.get(side) if isinstance(cache, dict) else {}
            if not isinstance(side_stats, dict):
                continue
            chunks.append(
                (
                    f"{side}("
                    f"dec={int(side_stats.get('decode_frames', 0) or 0)},"
                    f"inf={int(side_stats.get('infer_frames', 0) or 0)},"
                    f"hit={int(side_stats.get('center_hits', 0) or 0)},"
                    f"avg={float(side_stats.get('avg_infer_ms', 0.0) or 0.0):.1f}ms,"
                    f"last={float(side_stats.get('last_infer_ms', 0.0) or 0.0):.1f}ms/"
                    f"{self._fmt_age(side_stats.get('last_infer_age', -1.0))},"
                    f"hit_age={self._fmt_age(side_stats.get('last_hit_age', -1.0))},"
                    f"cache={int((cache_stats or {}).get('entries', 0) or 0)},"
                    f"cache_age={self._fmt_age((cache_stats or {}).get('newest_age', -1.0))},"
                    f"open={int(side_stats.get('reader_open_count', 0) or 0)},"
                    f"reconn={int(side_stats.get('reader_reconnect_count', 0) or 0)},"
                    f"gap={self._fmt_age(side_stats.get('reader_last_frame_gap', -1.0))},"
                    f"reason={side_stats.get('reader_last_reconnect_reason') or side_stats.get('reader_last_open_reason') or '-'}"
                    ')'
                )
            )
        return ';'.join(chunks)

    def _wait_for_wheel_results_for_type5(self, track_id, track_state):
        if not isinstance(track_state, dict) or self.wheel_result_provider is None:
            return
        wait_seconds = float(self.wheel_bind_wait_seconds)
        if wait_seconds <= 0.0:
            return
        expected = self._expected_wheel_side_count()
        started = time.time()
        deadline = started + wait_seconds
        locked_count = self._locked_wheel_side_count(track_state)
        while locked_count < expected:
            now = time.time()
            if now >= deadline:
                break
            self._update_wheel_track_activity(track_id, track_state, frame_ts=now, active=True)
            self._update_track_wheel_results(track_id, track_state, frame_ts=now)
            locked_count = self._locked_wheel_side_count(track_state)
            if locked_count >= expected:
                break
            time.sleep(min(self.wheel_bind_wait_poll_seconds, max(0.0, deadline - time.time())))
        elapsed = time.time() - started
        locked = track_state.get('wheel_results_locked') if isinstance(track_state, dict) else {}
        locked_sides = ','.join(sorted(str(side) for side in (locked or {}).keys())) if isinstance(locked, dict) else ''
        stats_part = ''
        if locked_count < expected:
            stats_part = (
                f" stats={self._format_wheel_runtime_stats()} "
                f"npu=[{format_npu_status(snapshot_npu_status())}]"
            )
        print(
            f"[wheel-bind] track={track_id} action=type5_wait "
            f"elapsed={elapsed:.2f}s locked={locked_count}/{expected} "
            f"locked_sides={locked_sides or '-'} "
            f"timeout={wait_seconds:.2f}s"
            f"{stats_part}"
        )

    def _update_track_wheel_results(self, track_id, track_state, frame_ts=None):
        if not isinstance(track_state, dict):
            return
        if self.wheel_bind_require_active and not bool(track_state.get('wheel_active', False)):
            return
        provider = getattr(self, 'wheel_result_provider', None)
        if provider is None:
            return
        getter = getattr(provider, 'get_recent_result_entries', None)
        if not callable(getter):
            return
        claimer = getattr(provider, 'claim_result_entry', None)
        ref_ts = time.time() if frame_ts is None else float(frame_ts)
        try:
            items = getter(now_ts=ref_ts, reference_ts=ref_ts, track_id=track_id)
        except TypeError:
            try:
                items = getter(now_ts=ref_ts, reference_ts=ref_ts)
            except TypeError:
                try:
                    items = getter(now_ts=ref_ts)
                except TypeError:
                    items = getter()
        except Exception as exc:
            for line in self.log_throttler.record(
                key='wheel_results.track_update_error',
                message=f'[wheel] failed to update lifecycle wheel results: {exc}',
                now=time.time(),
                window_seconds=10.0,
            ):
                print(line)
            return
        if not isinstance(items, list):
            return
        locked = track_state.setdefault('wheel_results_locked', {})
        photo_seed_candidates = []
        for item in items:
            if not isinstance(item, dict):
                continue
            side = str(item.get('side') or '').strip().lower()
            if side not in ('left', 'right'):
                continue
            capture_time = str(item.get('captureTime') or '').strip()
            class_name = str(item.get('className') or '').strip()
            image_bytes = item.get('imageJpegBytes', b'') or b''
            if not capture_time or not class_name or not image_bytes:
                continue
            candidate = {
                'side': side,
                'captureTime': capture_time,
                'imageJpegBytes': image_bytes,
                'className': class_name,
                'score': float(item.get('score', 0.0) or 0.0),
                'centerDistance': float(item.get('centerDistance', 0.0) or 0.0),
                'capture_ts': float(item.get('capture_ts', ref_ts) or ref_ts),
                'entryId': int(item.get('entryId', 0) or 0),
            }
            allowed, reject_reason = self._wheel_candidate_allowed_for_track(track_state, candidate, ref_ts)
            if not allowed:
                self._log_wheel_bind_decision(
                    track_id=track_id,
                    action='reject',
                    candidate=candidate,
                    ref_ts=ref_ts,
                    reason=reject_reason,
                )
                continue
            current = locked.get(side)
            if not isinstance(current, dict):
                if callable(claimer) and not claimer(track_id, candidate.get('entryId')):
                    continue
                locked[side] = candidate
                self._log_wheel_bind_decision(
                    track_id=track_id,
                    action='lock',
                    candidate=candidate,
                    ref_ts=ref_ts,
                )
                photo_seed_candidates.append(candidate)
                continue
            current_key = (
                -float(current.get('score', 0.0) or 0.0),
                -float(current.get('capture_ts', 0.0) or 0.0),
            )
            candidate_key = (
                -candidate['score'],
                -candidate['capture_ts'],
            )
            if candidate_key < current_key:
                if callable(claimer) and not claimer(track_id, candidate.get('entryId')):
                    continue
                locked[side] = candidate
                self._log_wheel_bind_decision(
                    track_id=track_id,
                    action='upgrade',
                    candidate=candidate,
                    ref_ts=ref_ts,
                )
                photo_seed_candidates.append(candidate)
            elif self._is_candidate_claimed_by_track(provider, track_id, candidate.get('entryId')):
                photo_seed_candidates.append(candidate)
        self._update_track_wheel_photo_history_from_provider(
            track_id=track_id,
            track_state=track_state,
            ref_ts=ref_ts,
            fallback_candidates=photo_seed_candidates,
        )

    @staticmethod
    def _is_candidate_claimed_by_track(provider, track_id, entry_id):
        entry_id = int(entry_id or 0)
        if entry_id <= 0:
            return False
        cache_entries = getattr(provider, '_entries', None)
        if not isinstance(cache_entries, dict):
            return False
        track_id = int(track_id or 0)
        for entries in cache_entries.values():
            for entry in entries or []:
                if int(entry.get('entryId', 0) or 0) != entry_id:
                    continue
                return int(entry.get('claimedTrackId', 0) or 0) == track_id
        return False

    def _get_claimed_wheel_entries(self, provider, track_id, ref_ts):
        getter = getattr(provider, 'get_claimed_result_entries', None)
        if not callable(getter):
            return []
        try:
            items = getter(track_id=track_id, now_ts=ref_ts, reference_ts=ref_ts)
        except TypeError:
            try:
                items = getter(track_id, now_ts=ref_ts, reference_ts=ref_ts)
            except TypeError:
                try:
                    items = getter(track_id, now_ts=ref_ts)
                except TypeError:
                    items = getter(track_id)
        except Exception as exc:
            for line in self.log_throttler.record(
                key='wheel_results.claimed_update_error',
                message=f'[wheel] failed to collect claimed wheel photo entries: {exc}',
                now=time.time(),
                window_seconds=10.0,
            ):
                print(line)
            return []
        if isinstance(items, list):
            return items
        return []

    def _get_photo_candidate_wheel_entries(self, provider, track_id, ref_ts):
        getter = getattr(provider, 'get_photo_candidate_entries', None)
        if not callable(getter):
            return []
        try:
            items = getter(track_id=track_id, now_ts=ref_ts, reference_ts=ref_ts)
        except TypeError:
            try:
                items = getter(track_id, now_ts=ref_ts, reference_ts=ref_ts)
            except TypeError:
                try:
                    items = getter(track_id, now_ts=ref_ts)
                except TypeError:
                    items = getter(track_id)
        except Exception as exc:
            for line in self.log_throttler.record(
                key='wheel_results.photo_candidate_error',
                message=f'[wheel] failed to collect wheel photo candidates: {exc}',
                now=time.time(),
                window_seconds=10.0,
            ):
                print(line)
            return []
        if isinstance(items, list):
            return items
        return []

    def _claim_photo_candidate_entries(self, track_id, candidates, claimer):
        if not callable(claimer):
            return
        for item in candidates or []:
            if not isinstance(item, dict):
                continue
            entry_id = int(item.get('entryId', 0) or 0)
            if entry_id <= 0:
                continue
            try:
                claimer(track_id, entry_id)
            except Exception:
                continue

    def _normalize_wheel_photo_candidate(self, item, ref_ts):
        if not isinstance(item, dict):
            return None
        side = str(item.get('side') or '').strip().lower()
        if side not in ('left', 'right'):
            return None
        capture_time = str(item.get('captureTime') or '').strip()
        class_name = str(item.get('className') or '').strip()
        image_bytes = item.get('imageJpegBytes', b'') or b''
        if not capture_time or not class_name or not image_bytes:
            return None
        return {
            'side': side,
            'captureTime': capture_time,
            'imageJpegBytes': image_bytes,
            'className': class_name,
            'score': float(item.get('score', 0.0) or 0.0),
            'centerDistance': float(item.get('centerDistance', 0.0) or 0.0),
            'capture_ts': float(item.get('capture_ts', ref_ts) or ref_ts),
            'entryId': int(item.get('entryId', 0) or 0),
        }

    def _update_track_wheel_photo_history_from_provider(
        self,
        track_id,
        track_state,
        ref_ts,
        fallback_candidates=None,
    ):
        if self.wheel_photo_uploader is None:
            return
        provider = getattr(self, 'wheel_result_provider', None)
        claimer = getattr(provider, 'claim_result_entry', None) if provider is not None else None
        photo_candidates = (
            self._get_photo_candidate_wheel_entries(provider, track_id, ref_ts)
            if provider is not None else []
        )
        self._claim_photo_candidate_entries(track_id, photo_candidates, claimer)
        raw_candidates = self._get_claimed_wheel_entries(provider, track_id, ref_ts) if provider is not None else []
        if not raw_candidates:
            raw_candidates = list(fallback_candidates or [])
        using_fallback = raw_candidates is fallback_candidates
        seen = set()
        for raw in raw_candidates:
            candidate = self._normalize_wheel_photo_candidate(raw, ref_ts)
            if not candidate:
                continue
            entry_id = int(candidate.get('entryId', 0) or 0)
            key = (candidate.get('side'), entry_id)
            if entry_id > 0 and key in seen:
                continue
            if entry_id > 0:
                seen.add(key)
            self._update_wheel_photo_history(
                track_id,
                track_state,
                candidate['side'],
                candidate,
                claimer=claimer if using_fallback else None,
            )

    def _update_wheel_photo_history(self, track_id, track_state, side, candidate, claimer=None):
        if self.wheel_photo_uploader is None:
            return
        if candidate.get('score', 0.0) < self.wheel_photo_min_score:
            return
        entry_id = int(candidate.get('entryId', 0) or 0)
        if callable(claimer):
            try:
                if not claimer(track_id, entry_id):
                    return
            except Exception:
                return
        history = track_state.setdefault('wheel_photo_history', {'left': {}, 'right': {}})
        side_history = history.setdefault(side, {})
        image_bytes = candidate.get('imageJpegBytes', b'') or b''
        image_hash = hashlib.sha1(image_bytes).hexdigest() if image_bytes else ''
        bucket_ts = int(candidate['capture_ts'] // self.wheel_photo_bucket_seconds)
        bucket = side_history.setdefault(bucket_ts, {'candidates': [], 'representative': None})
        if entry_id > 0 and any(int(c.get('entryId', 0) or 0) == entry_id for c in bucket['candidates']):
            return
        bucket['candidates'].append({
            'className': candidate.get('className', ''),
            'score': float(candidate.get('score', 0.0) or 0.0),
            'centerDistance': float(candidate.get('centerDistance', 0.0) or 0.0),
            'imageJpegBytes': image_bytes,
            'imageHash': image_hash,
            'capture_ts': float(candidate.get('capture_ts', 0.0) or 0.0),
            'entryId': entry_id,
        })
        new_rep_candidate = self._select_bucket_representative(bucket['candidates'])
        if new_rep_candidate is None:
            return
        clean_class_name = self._select_bucket_clean_class_name(bucket['candidates'])
        clean_value = WHEEL_CLASS_NAME_TO_CLEAN_VALUE.get(clean_class_name, 0)
        current_rep = bucket.get('representative')
        new_rep_hash = str(new_rep_candidate.get('imageHash') or '')
        if current_rep and int(current_rep.get('entryId', 0) or 0) == int(new_rep_candidate.get('entryId', 0) or 0):
            if int(current_rep.get('cleanValue', 0) or 0) != int(clean_value):
                current_rep['cleanValue'] = int(clean_value)
            return
        if current_rep and new_rep_hash and current_rep.get('imageHash') == new_rep_hash:
            current_rep.update({
                'cleanValue': int(clean_value),
                'score': float(new_rep_candidate.get('score', 0.0) or 0.0),
                'capture_ts': float(new_rep_candidate.get('capture_ts', 0.0) or 0.0),
                'entryId': int(new_rep_candidate.get('entryId', 0) or 0),
            })
            return
        duplicate_photo_url = ''
        duplicate_rep = None
        if new_rep_hash:
            for existing_bucket_key, existing_bucket in side_history.items():
                if existing_bucket_key == bucket_ts:
                    continue
                existing_rep = existing_bucket.get('representative') if isinstance(existing_bucket, dict) else None
                if (
                    isinstance(existing_rep, dict)
                    and existing_rep.get('imageHash') == new_rep_hash
                    and int(existing_rep.get('cleanValue', 0) or 0) == int(clean_value)
                ):
                    duplicate_photo_url = str(existing_rep.get('photoUrl') or '')
                    duplicate_rep = existing_rep
                    break
        seq_map = track_state.setdefault('wheel_photo_seq', {'left': 0, 'right': 0})
        if duplicate_rep and duplicate_rep.get('seq'):
            seq = int(duplicate_rep['seq'])
        elif current_rep and current_rep.get('seq'):
            seq = int(current_rep['seq'])
        else:
            seq = int(seq_map.get(side, 0)) + 1
            seq_map[side] = seq
        if duplicate_photo_url:
            photo_url = self._absolute_wheel_photo_url(duplicate_photo_url)
        else:
            photo_url = self._save_wheel_photo(
                side=side, track_id=track_id, seq=seq,
                image_bytes=new_rep_candidate.get('imageJpegBytes', b'') or b'',
                capture_ts=new_rep_candidate.get('capture_ts'),
                class_name=new_rep_candidate.get('className', ''),
            )
        if not photo_url:
            return
        bucket['representative'] = {
            'photoUrl': photo_url,
            'type': WHEEL_SIDE_TO_PHOTO_TYPE.get(side, ''),
            'cleanValue': clean_value,
            'score': float(new_rep_candidate.get('score', 0.0) or 0.0),
            'capture_ts': float(new_rep_candidate.get('capture_ts', 0.0) or 0.0),
            'entryId': int(new_rep_candidate.get('entryId', 0) or 0),
            'seq': seq,
            'imageHash': new_rep_hash,
            'duplicatePhoto': bool(duplicate_photo_url),
        }

    @staticmethod
    def _select_bucket_clean_class_name(candidates):
        if not candidates:
            return None
        counts = {}
        for c in candidates:
            cn = str(c.get('className', '') or '')
            counts[cn] = counts.get(cn, 0) + 1

        def type_rank(name):
            order = WHEEL_CLASS_NAME_TO_CLEAN_VALUE.get(name, 99)
            return (order if order else 99)

        best_type = min(counts.keys(), key=lambda cn: (-counts[cn], type_rank(cn)))
        return best_type

    @staticmethod
    def _select_bucket_representative(candidates):
        if not candidates:
            return None
        return min(candidates, key=lambda c: float(c.get('centerDistance', 0.0) or 0.0))

    def _enqueue_wheel_photos(self, track_state):
        if not self.wheel_photo_uploader:
            return
        history = track_state.get('wheel_photo_history') if isinstance(track_state, dict) else None
        if not isinstance(history, dict):
            return
        photos = []
        for side in ('left', 'right'):
            side_history = history.get(side)
            if not isinstance(side_history, dict):
                continue
            for bucket in side_history.values():
                if not isinstance(bucket, dict):
                    continue
                rep = bucket.get('representative')
                if not isinstance(rep, dict):
                    continue
                if rep.get('duplicatePhoto'):
                    continue
                photos.append((float(rep.get('capture_ts', 0.0) or 0.0), rep))
        photos.sort(key=lambda item: item[0])
        for _, entry in photos:
            photo_url = self._absolute_wheel_photo_url(entry.get('photoUrl'))
            type_str = str(entry.get('type') or '').strip()
            clean_value = entry.get('cleanValue')
            if not photo_url or not type_str or clean_value is None:
                continue
            try:
                clean_value_int = int(clean_value)
            except (TypeError, ValueError):
                continue
            try:
                self.wheel_photo_uploader.enqueue({
                    'photoUrl': photo_url,
                    'type': type_str,
                    'cleanValue': clean_value_int,
                })
            except Exception as exc:
                print(f'[wheel-photo] failed to enqueue ({photo_url}): {exc}')

    def _resolve_direction(self, track_state):
        state = None
        if track_state:
            state = track_state.get('zone_state')
        return self.zone_mgr.resolve_direction(state)

    def water_contact(self, box, water_boxes):
        if not water_boxes or box is None:
            return False
        for wb in water_boxes:
            if box_iou(box, wb) >= 0.02:
                return True
        return False
