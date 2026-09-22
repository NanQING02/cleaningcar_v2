import base64
import csv
import hashlib
import json
import re
import threading
import time
import urllib.request
from collections import OrderedDict, deque
from copy import deepcopy
from datetime import datetime
from math import ceil, hypot
from pathlib import Path

import cv2

from utils.upload_queue import SQLiteUploadQueue

from .constants import VEHICLE_LABEL_CN, WHEEL_CLASS_NAME_TO_CLEAN_VALUE, WHEEL_SIDE_TO_PHOTO_TYPE
from .event_trace import EventTraceRecorder
from .log_throttle import WindowedLogThrottler
from .lifecycle import BusinessLifecycleManager
from .npu_monitor import format_npu_status, snapshot_npu_status
from .plate import is_valid_plate, normalize_plate_candidate_text, normalize_plate_text
from .resize_accel import resize_bgr
from .vision import box_iou, get_anchor_point

PLATE_RECOGNITION_ABNORMAL_REASONS = frozenset({"PLATE_MISSING", "PLATE_NOT_LOCKED", "PLATE_NOT_DETECTED"})


def _format_track_debug_text(car_id, info):
    info = info or {}
    return (
        f"ID:{car_id} {info.get('state', 'idle')} "
        f"water:{'Y' if info.get('water', False) else 'N'} "
        f"wf:{int(info.get('water_consecutive_frames', 0) or 0)} "
        f"dur:{float(info.get('wash_duration', 0.0) or 0.0):.1f} "
        f"zb:{int(info.get('zone_b_elapsed', 0) or 0)}"
    )


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
                    error_text = f'{type(exc).__name__}: {exc}'
                    dead_letter_id = None
                    if self.db:
                        dead_letter_id = self.db.move_to_dead_letter(
                            job_id,
                            error_text,
                            retries=retries + 1,
                        )
                    print(
                        f'[uploader] moved event to dead-letter after {retries + 1} attempts '
                        f'dead_letter_id={dead_letter_id}: {error_text}'
                    )
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
                    error_text = f'{type(exc).__name__}: {exc}'
                    dead_letter_id = None
                    if self.db:
                        dead_letter_id = self.db.move_to_dead_letter(
                            job_id,
                            error_text,
                            retries=retries + 1,
                        )
                    print(
                        f'[wheel-photo-uploader] moved photo to dead-letter after {retries + 1} attempts '
                        f'dead_letter_id={dead_letter_id}: {error_text}'
                    )
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
        wheel_photo_bucket_seconds=0.5,
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
        self.evidence_index_enabled = bool(self.enable_event_disk or uploader)
        self._evidence_lock = threading.Lock()
        self.evidence_root = self.events_dir / 'evidence'
        self.evidence_index_path = self.events_dir / 'event_index.csv'
        if self.evidence_index_enabled:
            self.events_dir.mkdir(parents=True, exist_ok=True)
            self.evidence_root.mkdir(parents=True, exist_ok=True)
            if not self.evidence_index_path.exists():
                with self.evidence_index_path.open('w', encoding='utf-8-sig', newline='') as handle:
                    csv.writer(handle).writerow([
                        'capture_time', 'event_id', 'type', 'track_id', 'plate_number',
                        'plate_color', 'vehicle_type', 'lane', 'is_abnormal',
                        'abnormal_reason', 'capture_image', 'event_json', 'video_path', 'manifest',
                    ])
        self.tracks = {}
        self.timeout_frames = int(config.get('track_timeout_frames', 60))
        self.base_time = datetime.now()
        self.water_confirm_frames = max(1, int(self.logic.get('water_confirm_frames', 3)))
        shadow_cfg = dict(self.logic.get('shadow_plate_pool') or {})
        legacy_shadow_cfg = config.get('shadow_pool', {}) or {}
        for key, value in legacy_shadow_cfg.items():
            shadow_cfg.setdefault(key, value)
        self.shadow_max = int(shadow_cfg.get('max_candidates', 50))
        self.shadow_max_age = int(shadow_cfg.get('max_age_frames', 120))
        self.event_plate_lock_frames = max(
            1,
            int(self.logic.get('event_plate_lock_frames', self.logic.get('plate_lock_frames', 6))),
        )
        self.plate_text_max_streak_gap_frames = max(
            1,
            int(self.logic.get('plate_text_max_streak_gap_frames', 2)),
        )
        self.plate_correction_confirm_hits = max(
            self.event_plate_lock_frames,
            int(self.logic.get('plate_correction_confirm_hits', 12)),
        )
        self.plate_text_window_frames = max(
            self.event_plate_lock_frames,
            int(shadow_cfg.get('text_window_frames', min(self.shadow_max_age, 50))),
        )
        self.plate_text_margin_ratio = float(shadow_cfg.get('text_margin_ratio', 0.12))
        self.plate_text_switch_min_consecutive = max(
            self.event_plate_lock_frames,
            int(shadow_cfg.get('text_switch_min_consecutive', 6)),
        )
        self.plate_text_switch_gain_ratio = float(shadow_cfg.get('text_switch_gain_ratio', 1.2))
        self.plate_text_switch_margin_ratio = float(
            shadow_cfg.get('text_switch_margin_ratio', max(self.plate_text_margin_ratio + 0.05, 0.18))
        )
        self.plate_text_min_detection_confidence = float(
            self.logic.get(
                'plate_text_min_detection_confidence',
                shadow_cfg.get('text_min_detection_confidence', 0.65),
            )
        )
        self.plate_text_min_recognition_confidence = float(
            self.logic.get(
                'plate_text_min_recognition_confidence',
                shadow_cfg.get('text_min_recognition_confidence', 0.75),
            )
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
        self.plate_color_correction_hits = max(
            self.plate_color_lock_frames,
            int(self.logic.get('plate_color_correction_hits', 5)),
        )
        self.plate_yellow_green_fusion_enabled = bool(
            self.logic.get('plate_yellow_green_fusion_enabled', True)
        )
        self.plate_yellow_green_min_confidence = max(
            0.0,
            min(1.0, float(self.logic.get('plate_yellow_green_min_confidence', 0.55))),
        )
        self.plate_yellow_green_window_frames = max(
            1,
            int(self.logic.get('plate_yellow_green_window_frames', self.plate_color_window_frames)),
        )
        self.plate_yellow_green_min_hits_per_color = max(
            1,
            int(self.logic.get('plate_yellow_green_min_hits_per_color', 2)),
        )
        configured_plate_colors = self.logic.get('allowed_plate_colors', ['蓝色', '黄色', '绿色', '黄绿色'])
        if not isinstance(configured_plate_colors, (list, tuple, set)):
            configured_plate_colors = ['蓝色', '黄色', '绿色', '黄绿色']
        self.allowed_plate_colors = {
            str(color).strip()
            for color in configured_plate_colors
            if str(color).strip()
        }
        if self.plate_yellow_green_fusion_enabled:
            self.allowed_plate_colors.add('黄绿色')
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
        self.lifecycle_manager = BusinessLifecycleManager(
            self.camera_id,
            grace_seconds=float(self.logic.get('track_lost_grace_seconds', 4.0) or 4.0),
        )
        self.track_lost_grace_seconds = max(
            0.0,
            float(self.logic.get('track_lost_grace_seconds', 4.0) or 4.0),
        )
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
        self.min_type1_track_frames = int(self.logic.get('min_track_frames_for_type1', 5))
        if self.min_type1_track_frames < 0:
            self.min_type1_track_frames = 0
        self.wash_dwell_offset = 0.0
        self.min_type4_zone_b_dwell_seconds = max(
            0.0,
            float(self.logic.get('min_zone_b_dwell_seconds_for_type4', 0.5)),
        )
        self.min_type4_zone_b_dwell = max(
            1,
            int(ceil(max(self.fps, 1.0) * self.min_type4_zone_b_dwell_seconds)),
        )
        self.allowed_events = {1, 2, 3, 4, 5, 6}
        self.disable_plate_only_events = True
        self.single_lifecycle_events = True
        self.pre_type2_video_segment_seconds = max(
            0.0,
            float(self.logic.get('pre_type2_video_segment_seconds', 600.0) or 0.0),
        )
        self.post_type2_force_finalize_seconds = max(
            0.0,
            float(self.logic.get('post_type2_force_finalize_seconds', 900.0) or 0.0),
        )
        self.per_id_video_tail_seconds = 8.0
        self.per_id_video_enabled = bool(self.logic.get('enable_per_id_video', False))
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
        self.frame_timing_max_entries = max(
            128,
            min(10000, int(self.logic.get('frame_timing_max_entries', 4096) or 4096)),
        )
        self.frame_timing = OrderedDict()
        self.log_throttler = WindowedLogThrottler()
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
        self.event_trace = EventTraceRecorder.from_config(config, fps)

    def trace_record(self, kind, payload=None):
        recorder = getattr(self, 'event_trace', None)
        if recorder is None:
            return False
        return recorder.record(kind, payload)

    def close(self):
        recorder = getattr(self, 'event_trace', None)
        if recorder is not None:
            recorder.close()

    @staticmethod
    def _safe_evidence_component(value):
        text = re.sub(r'[^\w.\-\u4e00-\u9fff]+', '_', str(value or '').strip(), flags=re.UNICODE)
        return text.strip('._')[:96] or 'unknown-event'

    def _record_event_evidence(self, event, track_state, frame_idx, event_path=None):
        if not self.evidence_index_enabled or not isinstance(event, dict):
            return None
        event_id = str(event.get('id') or '').strip()
        if not event_id:
            return None
        capture_time = str(event.get('captureTime') or '')
        date_key = re.sub(r'\D', '', capture_time)[:8]
        if len(date_key) != 8:
            date_key = datetime.now().strftime('%Y%m%d')
        event_dir = self.evidence_root / date_key / self._safe_evidence_component(event_id)
        manifest_path = event_dir / 'manifest.json'
        event_type = int(event.get('type', 0) or 0)
        stage_labels = {
            1: '进入检测区',
            2: '进入冲洗区',
            3: '检测到冲洗水流',
            4: '离开冲洗区',
            5: '车辆业务闭环',
            6: '录像落盘完成',
        }
        upload_state = 'disabled'
        if self.uploader:
            upload_state = 'waiting_type2' if event_type == 1 else 'queued'
        stage = {
            'stageKey': f'{event_type}:{int(frame_idx)}',
            'type': event_type,
            'label': stage_labels.get(event_type, f'type{event_type}'),
            'captureTime': capture_time,
            'frameIdx': int(frame_idx),
            'trackId': int(event.get('trackId', 0) or 0),
            'plateNumber': str(event.get('plateNumber') or ''),
            'plateColor': str(event.get('plateColor') or ''),
            'vehicleType': str(event.get('vehicleType') or ''),
            'isAbnormal': bool(event.get('isAbnormal', False)),
            'abnormalReason': str(event.get('abnormalReason') or ''),
            'sequenceBackfill': bool(event.get('sequenceBackfill', False)),
            'backfillReason': str(event.get('backfillReason') or ''),
            'captureImage': str(event.get('captureImage') or ''),
            'eventJson': str(event_path or ''),
            'videoPath': str((track_state or {}).get('per_id_video_path') or ''),
            'uploadState': upload_state,
        }
        with self._evidence_lock:
            event_dir.mkdir(parents=True, exist_ok=True)
            manifest = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                except Exception:
                    manifest = {}
            stages = list(manifest.get('stages') or [])
            stages = [item for item in stages if item.get('stageKey') != stage['stageKey']]
            stages.append(stage)
            stages.sort(key=lambda item: (
                str(item.get('captureTime') or ''),
                int(item.get('frameIdx', 0) or 0),
                int(item.get('type', 0) or 0),
            ))
            for order, item in enumerate(stages, start=1):
                item['order'] = order
            capture_images = list(dict.fromkeys(
                str(item.get('captureImage') or '') for item in stages if item.get('captureImage')
            ))
            event_json_files = list(dict.fromkeys(
                str(item.get('eventJson') or '') for item in stages if item.get('eventJson')
            ))
            video_paths = list(dict.fromkeys(
                str(item.get('videoPath') or '') for item in stages if item.get('videoPath')
            ))
            stage_types = [int(item.get('type', 0) or 0) for item in stages]
            manifest.update({
                'schemaVersion': 1,
                'eventId': event_id,
                'deviceId': self.camera_id,
                'lane': str(event.get('lane') or ''),
                'firstCaptureTime': stages[0].get('captureTime') if stages else capture_time,
                'lastCaptureTime': stages[-1].get('captureTime') if stages else capture_time,
                'currentType': max(stage_types) if stage_types else event_type,
                'businessComplete': 5 in stage_types,
                'videoComplete': 6 in stage_types,
                'plateNumber': str(event.get('plateNumber') or manifest.get('plateNumber') or ''),
                'plateColor': str(event.get('plateColor') or manifest.get('plateColor') or ''),
                'vehicleType': str(event.get('vehicleType') or manifest.get('vehicleType') or ''),
                'isAbnormal': any(bool(item.get('isAbnormal')) for item in stages),
                'abnormalReasons': sorted({
                    str(item.get('abnormalReason') or '') for item in stages if item.get('abnormalReason')
                }),
                'stages': stages,
                'artifacts': {
                    'captureImages': capture_images,
                    'eventJsonFiles': event_json_files,
                    'videoPaths': video_paths,
                },
            })
            temp_path = manifest_path.with_suffix('.json.tmp')
            temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            temp_path.replace(manifest_path)
            with self.evidence_index_path.open('a', encoding='utf-8', newline='') as handle:
                csv.writer(handle).writerow([
                    capture_time,
                    event_id,
                    event_type,
                    stage['trackId'],
                    stage['plateNumber'],
                    stage['plateColor'],
                    stage['vehicleType'],
                    str(event.get('lane') or ''),
                    int(stage['isAbnormal']),
                    stage['abnormalReason'],
                    stage['captureImage'],
                    stage['eventJson'],
                    stage['videoPath'],
                    str(manifest_path),
                ])
        return manifest_path

    def record_frame_timing(self, frame_idx, capture_ts, infer_ts):
        if capture_ts is None or infer_ts is None:
            return
        try:
            normalized_frame = int(frame_idx)
            self.frame_timing[normalized_frame] = (float(capture_ts), float(infer_ts))
            self.frame_timing.move_to_end(normalized_frame)
            while len(self.frame_timing) > self.frame_timing_max_entries:
                self.frame_timing.popitem(last=False)
        except Exception:
            return

    def _capture_timestamp(self, frame_idx):
        timing = self.frame_timing.get(int(frame_idx))
        return timing[0] if timing else None

    def _lifecycle_zone_edge(self, anchor_point, plate_box):
        point = anchor_point
        if point is None and plate_box is not None and len(plate_box) >= 4:
            point = (
                0.5 * (float(plate_box[0]) + float(plate_box[2])),
                0.5 * (float(plate_box[1]) + float(plate_box[3])),
            )
        if point is None:
            return ''
        try:
            ratio = float(self.zone_mgr._relative_position(point))
        except (AttributeError, TypeError, ValueError):
            return ''
        return 'flow_start' if ratio <= 0.5 else 'flow_end'

    @staticmethod
    def _plate_vehicle_motion_consistent(previous_vehicle_box, previous_plate_box, vehicle_box, plate_box):
        if previous_vehicle_box is None or previous_plate_box is None or vehicle_box is None or plate_box is None:
            return True
        previous_vehicle_center = (
            0.5 * (previous_vehicle_box[0] + previous_vehicle_box[2]),
            0.5 * (previous_vehicle_box[1] + previous_vehicle_box[3]),
        )
        vehicle_center = (0.5 * (vehicle_box[0] + vehicle_box[2]), 0.5 * (vehicle_box[1] + vehicle_box[3]))
        previous_plate_center = (
            0.5 * (previous_plate_box[0] + previous_plate_box[2]),
            0.5 * (previous_plate_box[1] + previous_plate_box[3]),
        )
        plate_center = (0.5 * (plate_box[0] + plate_box[2]), 0.5 * (plate_box[1] + plate_box[3]))
        vehicle_delta = (
            vehicle_center[0] - previous_vehicle_center[0],
            vehicle_center[1] - previous_vehicle_center[1],
        )
        plate_delta = (
            plate_center[0] - previous_plate_center[0],
            plate_center[1] - previous_plate_center[1],
        )
        vehicle_speed = hypot(*vehicle_delta)
        plate_speed = hypot(*plate_delta)
        if vehicle_speed >= 8.0 and plate_speed >= 4.0:
            cosine = (vehicle_delta[0] * plate_delta[0] + vehicle_delta[1] * plate_delta[1]) / max(
                vehicle_speed * plate_speed, 1e-6
            )
            if cosine < 0.0:
                return False
        vehicle_width = max(float(vehicle_box[2]) - float(vehicle_box[0]), 1.0)
        vehicle_height = max(float(vehicle_box[3]) - float(vehicle_box[1]), 1.0)
        previous_width = max(float(previous_vehicle_box[2]) - float(previous_vehicle_box[0]), 1.0)
        previous_height = max(float(previous_vehicle_box[3]) - float(previous_vehicle_box[1]), 1.0)
        previous_relative = (
            (previous_plate_center[0] - previous_vehicle_box[0]) / previous_width,
            (previous_plate_center[1] - previous_vehicle_box[1]) / previous_height,
        )
        relative = (
            (plate_center[0] - vehicle_box[0]) / vehicle_width,
            (plate_center[1] - vehicle_box[1]) / vehicle_height,
        )
        return hypot(relative[0] - previous_relative[0], relative[1] - previous_relative[1]) <= 0.25

    def _observe_lifecycle(self, track_id, track_state, frame_idx, vehicle_label,
                           plate_box, anchor_point, plate_is_guess, anchor_direction):
        capture_ts = self._capture_timestamp(frame_idx)
        if capture_ts is None:
            return None
        vehicle_class = str(track_state.get('vehicle_cls_locked') or track_state.get('vehicle_cls') or vehicle_label or '')
        locked_plate = normalize_plate_candidate_text(track_state.get('plate_text_locked') or '')
        has_valid_plate = bool(locked_plate and not track_state.get('plate_text_locked_is_guess') and not plate_is_guess)
        locked_color = str(track_state.get('plate_color_locked') or '')
        locked_color_conf = float(track_state.get('plate_color_locked_conf', 0.0) or 0.0)
        plate_type = str(track_state.get('plate_type') or '')
        motion_direction = str((anchor_direction or {}).get('motion_direction', 'unknown') or 'unknown')
        handoff_plate_box = plate_box or track_state.get('last_plate_box')
        plate_edge = self._lifecycle_zone_edge(anchor_point, handoff_plate_box)
        lifecycle = self.lifecycle_manager.get(track_id)
        waiting_lifecycles = []
        if lifecycle is None and vehicle_class:
            waiting_lifecycles = self.lifecycle_manager.find_waiting_lifecycles(vehicle_class, capture_ts)
            track_state['lifecycle_handoff_pending'] = bool(
                track_state.get('born_inside_a') and waiting_lifecycles and not has_valid_plate
            )
        if lifecycle is None and has_valid_plate and vehicle_class:
            candidates = self.lifecycle_manager.find_handoff_candidates(
                vehicle_class,
                capture_ts,
                locked_plate,
                plate_edge,
                handoff_plate_box,
                has_valid_plate=True,
                motion_direction=motion_direction,
            )
            if len(candidates) == 1:
                lifecycle = candidates[0]
                from_track_id = int(lifecycle.last_tracker_id or 0)
                previous_state = self.tracks.get(from_track_id) or {}
                lifecycle = self.lifecycle_manager.handoff(lifecycle, track_id, capture_ts)
                if lifecycle is not None:
                    track_state['lifecycle_handoff_pending'] = False
                    track_state['session_id'] = lifecycle.event_id
                    track_state['_lifecycle_handoff_from'] = int(from_track_id)
                    track_state['events'] = set(lifecycle.stages)
                    track_state['event_stage_max'] = max(lifecycle.stages) if lifecycle.stages else 0
                    for field in ('record_start_frame', 'record_stop_frame', 'type1_capture_time'):
                        if field in previous_state:
                            track_state[field] = previous_state[field]
                    for field in (
                        'record_first_start_frame',
                        'record_segment_start_frame',
                        'pre_type2_rotate_requested',
                        'pre_type2_rotation_count',
                        'recording_committed',
                        'per_id_recording_ready',
                        'per_id_video_path',
                    ):
                        if field in previous_state:
                            track_state[field] = previous_state[field]
                    for field in ('class_counts', 'wash_stage_class_counts'):
                        if isinstance(previous_state.get(field), dict):
                            track_state[field] = dict(previous_state[field])
                    for field in (
                        'vehicle_cls',
                        'vehicle_cls_locked',
                        'vehicle_cls_frozen',
                        'vehicle_cls_at_type2',
                        'vehicle_cls_lock_reason',
                    ):
                        if field in previous_state:
                            track_state[field] = previous_state[field]
                    track_state['plate_text_locked'] = lifecycle.last_plate_text
                    track_state['plate_text_locked_is_guess'] = False
                    track_state['plate_text'] = lifecycle.last_plate_text
                    track_state['plate_is_guess'] = False
                    track_state['plate_color_locked'] = lifecycle.last_plate_color
                    track_state['plate_color_locked_conf'] = lifecycle.last_plate_color_conf
                    track_state['plate_color_locked_text'] = lifecycle.last_plate_text if lifecycle.last_plate_color else ''
                    track_state['plate_color'] = lifecycle.last_plate_color
                    track_state['plate_color_conf'] = lifecycle.last_plate_color_conf
                    for field in (
                        'plate_color_evidence_by_text',
                        'plate_color_fusion_evidence_by_text',
                        'plate_color_votes_by_text',
                    ):
                        if isinstance(previous_state.get(field), dict):
                            track_state[field] = deepcopy(previous_state[field])
                    track_state['plate_type'] = lifecycle.last_plate_type
                    track_state['type2_qualified'] = 2 in lifecycle.stages
                    previous_state['_lifecycle_superseded'] = True
                    self.trace_record('lifecycle_handoff', {
                        'frameIdx': int(frame_idx),
                        'trackId': int(track_id),
                        'eventId': lifecycle.event_id,
                        'fromTrackId': int(from_track_id),
                        'reason': 'unique_plate_verified_candidate',
                    })
            elif len(candidates) > 1:
                track_state['lifecycle_handoff_pending'] = True
                self.trace_record('lifecycle_handoff_rejected', {
                    'frameIdx': int(frame_idx),
                    'trackId': int(track_id),
                    'reason': 'ambiguous_candidates',
                    'candidateCount': len(candidates),
                })
            else:
                track_state['lifecycle_handoff_pending'] = False
                if waiting_lifecycles:
                    self.trace_record('lifecycle_handoff_rejected', {
                        'frameIdx': int(frame_idx),
                        'trackId': int(track_id),
                        'reason': 'stable_plate_mismatch_or_spatial_discontinuity',
                        'plateText': locked_plate,
                        'candidateCount': len(waiting_lifecycles),
                    })
        if lifecycle is not None:
            track_state['lifecycle_handoff_pending'] = False
            self.lifecycle_manager.touch(
                track_id,
                capture_ts=capture_ts,
                vehicle_class=vehicle_class,
                plate_text=locked_plate if has_valid_plate else '',
                plate_box=handoff_plate_box if has_valid_plate else None,
                plate_edge=plate_edge,
                motion_direction=motion_direction,
                plate_color=locked_color if has_valid_plate else '',
                plate_color_conf=locked_color_conf if has_valid_plate else 0.0,
                plate_type=plate_type if has_valid_plate else '',
            )
        return lifecycle

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
        text_conf=None,
        mutual_verified=False,
        motion_consistent=True,
        plate_color='',
        plate_color_conf=None,
        plate_type='',
        update_frame_idx=None,
        switch_trusted=None,
    ):
        normalized_plate = normalize_plate_candidate_text(text)
        has_valid_plate_candidate = bool(normalized_plate and is_valid_plate(normalized_plate))
        try:
            candidate_frame = int(frame_idx)
        except (TypeError, ValueError):
            candidate_frame = 0
        detection_confidence = float(conf) if conf is not None else None
        recognition_confidence = float(text_conf) if text_conf is not None else None
        candidate_eligible = True
        if detection_confidence is not None:
            candidate_eligible = candidate_eligible and detection_confidence >= self.plate_text_min_detection_confidence
        if recognition_confidence is not None:
            candidate_eligible = candidate_eligible and recognition_confidence >= self.plate_text_min_recognition_confidence
        candidate_trusted = bool(trusted and candidate_eligible)
        candidate_switch_trusted = bool(
            (trusted if switch_trusted is None else switch_trusted)
            and candidate_eligible
        )
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
                min(1.0, float(conf or 0.0) + (0.10 if mutual_verified else 0.0)),
                candidate_frame,
                trusted=candidate_trusted,
            )
            if candidate_trusted:
                self._update_locked_plate_text(
                    track_id,
                    track_state,
                    candidate_frame if update_frame_idx is None else update_frame_idx,
                    observed_text=normalized_plate,
                    observed_trusted=True,
                    observed_switch_trusted=candidate_switch_trusted,
                )
            else:
                self.trace_record('plate_text_evidence', {
                    'frameIdx': int(candidate_frame),
                    'trackId': int(track_id),
                    'text': normalized_plate,
                    'detectionConfidence': detection_confidence,
                    'recognitionConfidence': recognition_confidence,
                    'mutualVerified': bool(mutual_verified),
                    'motionConsistent': bool(motion_consistent),
                    'action': 'buffered',
                    'reason': 'below_trusted_lock_threshold',
                })
        if plate_color:
            parsed_color_conf = 0.0
            if plate_color_conf is not None:
                try:
                    parsed_color_conf = float(plate_color_conf)
                except (TypeError, ValueError):
                    parsed_color_conf = 0.0
            self._update_locked_plate_color(
                track_id,
                track_state,
                normalized_plate,
                str(plate_color).strip(),
                parsed_color_conf,
                candidate_frame,
                trusted=candidate_trusted,
            )
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
                trusted=False,
                switch_trusted=False,
                text_conf=entry.get('text_conf'),
                mutual_verified=entry.get('mutual_verified', False),
                motion_consistent=entry.get('motion_consistent', False),
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
                     plate_color='', plate_color_conf=None, plate_type='', plate_candidate_history=None,
                     anchor_direction=None, plate_text_conf=None, plate_mutual_verified=False,
                     plate_motion_consistent=True):
        if track_id <= 0:
            return
        if self.disable_plate_only_events and is_plate and vehicle_box is None:
            existing_state = self.tracks.get(track_id)
            if existing_state:
                existing_state['plate_initial_candidate'] = ''
                existing_state['plate_initial_streak'] = 0
                self._reset_plate_switch_streak(existing_state)
            return
        previous_state = self.tracks.get(track_id) or {}
        self.trace_record('track_input', {
            'frameIdx': int(frame_idx),
            'trackId': int(track_id),
            'isPlateUpdate': bool(is_plate),
            'plateBox': plate_box,
            'vehicleBox': vehicle_box,
            'plateText': str(plate_text or ''),
            'plateConfidence': plate_conf,
            'plateTextConfidence': plate_text_conf,
            'plateMutualVerified': bool(plate_mutual_verified),
            'plateMotionConsistent': bool(plate_motion_consistent),
            'plateColor': str(plate_color or ''),
            'plateColorConfidence': plate_color_conf,
            'vehicleLabel': str(vehicle_label or ''),
            'vehicleConfidence': vehicle_conf,
            'waterBoxes': water_boxes or [],
            'waterActive': bool(water_active),
            'confirmed': bool(confirmed),
            'cleaningLabel': str(cleaning_label or ''),
            'anchorPoint': anchor_point,
            'anchorDirection': anchor_direction or {},
            'previousEvents': sorted(previous_state.get('events', set())),
            'previousStage': int(previous_state.get('event_stage_max', 0) or 0),
            'previousLastFrame': previous_state.get('last_frame_idx'),
        })
        st = self.tracks.setdefault(track_id, {
            'track_id': track_id,
            'events': set(),
            'event_stage_max': 0,
            'event_sequence_issues': [],
            'wash_duration': 0.0,
            'washing': False,
            'washing_candidate': False,
            'washing_confirmed': False,
            'water_detected': False,
            'water_consecutive_frames': 0,
            'last_water_observation_frame': -1,
            'effective_wash_frames': 0,
            'last_frame_idx': frame_idx,
            'last_frame': None,
            'plate_text_latest': '',
            'plate_text_locked': '',
            'plate_text_locked_is_guess': False,
            'plate_initial_candidate': '',
            'plate_initial_streak': 0,
            'plate_text_switch_candidate': '',
            'plate_text_switch_streak': 0,
            'plate_text': '',
            'plate_is_guess': False,
            'plate_color_latest': '',
            'plate_color_latest_conf': 0.0,
            'plate_color_locked': '',
            'plate_color_locked_conf': 0.0,
            'plate_color_locked_text': '',
            'plate_color_votes': {},
            'plate_color_evidence_by_text': {},
            'plate_color_fusion_evidence_by_text': {},
            'plate_color_votes_by_text': {},
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
            'wash_start_time': None,
            'wash_end_time': None,
            'lane': self.lane_name,
            'last_cleaning': '',
            'vehicle_cls_frozen': False,
            'class_counts': {},
            'wash_stage_class_counts': {},
            'vehicle_cls_locked': '',
            'vehicle_cls_at_type2': '',
            'vehicle_cls_lock_reason': '',
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
            'born_inside_a': False,
            'born_inside_pending': False,
            'born_inside_outcome': '',
            'anchor_motion_direction': 'unknown',
            'anchor_direction_confidence': 0.0,
            'anchor_direction_locked': False,
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
            'wheel_active': False,
            'wheel_activity_start_ts': None,
            'wheel_activity_last_ts': None,
            'wheel_activity_end_ts': None,
            'lifecycle_handoff_pending': False,
            'deferred_type2': False,
            'record_first_start_frame': None,
            'record_segment_start_frame': None,
            'pre_type2_rotate_requested': False,
            'pre_type2_rotation_count': 0,
            'recording_committed': False,
            'per_id_recording_ready': False,
            'per_id_video_path': '',
        })
        if self.single_lifecycle_events and st.get('closed'):
            st['last_frame_idx'] = frame_idx
            self._update_wheel_track_activity(track_id, st, frame_ts=time.time(), active=False)
            return
        st['track_id'] = track_id
        st['track_frame_count'] = st.get('track_frame_count', 0) + 1
        st['last_frame_idx'] = frame_idx
        if isinstance(anchor_direction, dict):
            motion_direction = str(anchor_direction.get('motion_direction', 'unknown') or 'unknown')
            try:
                direction_confidence = float(anchor_direction.get('direction_confidence', 0.0) or 0.0)
            except (TypeError, ValueError):
                direction_confidence = 0.0
            direction_locked = bool(anchor_direction.get('direction_locked', False))
            if direction_locked and motion_direction in {'forward', 'reverse'}:
                st['anchor_motion_direction'] = motion_direction
                st['anchor_direction_confidence'] = max(0.0, min(1.0, direction_confidence))
                st['anchor_direction_locked'] = True
        if frame is not None:
            st['last_frame'] = frame.copy() if self.copy_track_last_frame else frame
        counts = st.get('class_counts') or {}
        if vehicle_label and not st.get('vehicle_cls_frozen'):
            counts[vehicle_label] = counts.get(vehicle_label, 0) + 1
            st['class_counts'] = counts
            if st.get('type2_qualified'):
                stage_counts = st.get('wash_stage_class_counts') or {}
                stage_counts[vehicle_label] = stage_counts.get(vehicle_label, 0) + 1
                st['wash_stage_class_counts'] = stage_counts
            st['vehicle_cls'] = self._select_vehicle_class(st)
        if vehicle_label:
            st['last_vehicle_label'] = vehicle_label
            if not st.get('vehicle_cls'):
                st['vehicle_cls'] = vehicle_label
        prev_vehicle_box_for_motion = st.get('last_vehicle_box')
        prev_plate_box_for_motion = st.get('last_plate_box')
        if vehicle_box is not None:
            st['last_vehicle_box'] = vehicle_box
        plate_motion_consistent = bool(
            plate_motion_consistent and self._plate_vehicle_motion_consistent(
                prev_vehicle_box_for_motion,
                prev_plate_box_for_motion,
                vehicle_box,
                plate_box,
            )
        )
        if plate_box is not None and plate_motion_consistent:
            st['last_plate_box'] = plate_box
        self._merge_plate_candidate_history(track_id, st, plate_candidate_history, frame_idx)
        self._ingest_plate_candidate(
            track_id,
            st,
            plate_text,
            frame_idx,
            conf=plate_conf,
            trusted=True,
            switch_trusted=True,
            text_conf=plate_text_conf,
            mutual_verified=plate_mutual_verified,
            motion_consistent=plate_motion_consistent,
            plate_color=plate_color,
            plate_color_conf=plate_color_conf,
            plate_type=plate_type,
        )
        self._sync_plate_legacy_fields(st)
        if confirmed:
            st['confirmed'] = True
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
        reliable_vehicle_box = vehicle_box or st.get('last_vehicle_box')
        vehicle_height = None
        if reliable_vehicle_box is not None and len(reliable_vehicle_box) >= 4:
            try:
                candidate_height = float(reliable_vehicle_box[3]) - float(reliable_vehicle_box[1])
                if candidate_height > 0.0:
                    vehicle_height = candidate_height
            except (TypeError, ValueError):
                vehicle_height = None
        zone_state, zone_flags = self.zone_mgr.update_track(
            track_id,
            anchor_point,
            frame_idx,
            vehicle_height=vehicle_height,
        )
        st['zone_state'] = zone_state
        born_inside_a = bool(getattr(zone_state, 'born_inside_a', False))
        born_inside_pending = bool(getattr(zone_state, 'born_inside_pending', False))
        born_inside_outcome = str(getattr(zone_state, 'born_inside_outcome', '') or '')
        if born_inside_a:
            st['born_inside_a'] = True
        st['born_inside_pending'] = born_inside_pending
        previous_born_inside_outcome = str(st.get('born_inside_outcome', '') or '')
        st['born_inside_outcome'] = born_inside_outcome
        if born_inside_outcome and born_inside_outcome != previous_born_inside_outcome:
            self.trace_record('born_inside', {
                'frameIdx': int(frame_idx),
                'trackId': int(track_id),
                'action': born_inside_outcome.lower(),
                'insideA': bool(getattr(zone_state, 'inside_a', False)),
                'insideB': bool(getattr(zone_state, 'inside_b', False)),
            })
        self.trace_record('zone_update', {
            'frameIdx': int(frame_idx),
            'trackId': int(track_id),
            'insideA': bool(zone_state and zone_state.inside_a),
            'insideB': bool(zone_state and zone_state.inside_b),
            'zoneBRegion': getattr(zone_state, 'zone_b_region', 'INVALID'),
            'zoneBSignedDistance': getattr(zone_state, 'zone_b_signed_distance', None),
            'zoneBDynamicMargin': getattr(zone_state, 'zone_b_dynamic_margin', 0.0),
            'flags': zone_flags,
            'anchorPoint': anchor_point,
            'zoneAState': getattr(zone_state, 'zone_a_state', 'UNSEEN'),
            'zoneARegion': getattr(zone_state, 'zone_a_region', 'INVALID'),
            'signedDistance': getattr(zone_state, 'signed_distance', None),
            'dynamicMargin': getattr(zone_state, 'dynamic_margin', 0.0),
            'observedOutsideCount': getattr(zone_state, 'observed_outside_count', 0),
            'enterCoreCount': getattr(zone_state, 'enter_core_count', 0),
            'exitOutsideCount': getattr(zone_state, 'exit_outside_count', 0),
            'initialCoreCompat': bool(getattr(zone_state, 'initial_core_compat', False)),
            'bornInsideA': born_inside_a,
            'bornInsidePending': born_inside_pending,
            'bornInsideOutcome': born_inside_outcome,
            'transitionReason': getattr(zone_state, 'transition_reason', ''),
        })
        self._observe_lifecycle(
            track_id,
            st,
            frame_idx,
            vehicle_label,
            plate_box,
            anchor_point,
            plate_is_guess,
            anchor_direction,
        )

        timestamp = self.frame_timestamp(frame_idx)
        inside_a = bool(zone_state and zone_state.inside_a)
        inside_b = bool(zone_state and zone_state.inside_b)
        if inside_a or zone_flags.get('enter_a') or st.get('zone_a_dwell_frames', 0) > 0:
            st['zone_a_seen'] = True
        if zone_flags.get('exit_a'):
            st['zone_a_exited'] = True
        if zone_flags.get('enter_a'):
            st['zone_a_exited'] = False
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
            st['water_consecutive_frames'] = 0
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
        if type2_ready and st.get('lifecycle_handoff_pending'):
            st['deferred_type2'] = True
            type2_ready = False
        elif st.get('deferred_type2') and inside_b and not st.get('lifecycle_handoff_pending'):
            type2_ready = True
            st['deferred_type2'] = False
        type2_newly_qualified = bool(type2_ready and not st.get('type2_qualified'))
        if type2_ready:
            st['type2_qualified'] = True
            if st.get('type2_qualified_frame', -1) < 0:
                st['type2_qualified_frame'] = frame_idx
            if type2_newly_qualified:
                st['recording_committed'] = True
                st['pre_type2_rotate_requested'] = False
                self._start_wash_stage_vehicle_class(st, vehicle_label)

        wheel_ref_ts = time.time()
        wheel_active_now = self._is_wheel_track_active(st)
        self._update_wheel_track_activity(track_id, st, frame_ts=wheel_ref_ts, active=wheel_active_now)
        if wheel_active_now or not self.wheel_bind_require_active:
            self._update_track_wheel_results(track_id, st, frame_ts=wheel_ref_ts)

        water_in_b = bool(st.get('type2_qualified') and inside_b and water_boxes)
        last_water_frame = st.get('last_water_observation_frame', -1)
        if int(last_water_frame if last_water_frame is not None else -1) != int(frame_idx):
            st['last_water_observation_frame'] = int(frame_idx)
            if water_in_b:
                st['water_consecutive_frames'] = int(st.get('water_consecutive_frames', 0) or 0) + 1
                st['effective_wash_frames'] = st.get('effective_wash_frames', 0) + 1
            else:
                st['water_consecutive_frames'] = 0
        washing_now = bool(water_in_b)

        if self.disable_plate_only_events and is_plate and (vehicle_box is None and st.get('last_vehicle_box') is None):
            return
        can_type1 = self._can_emit_type1(st)
        born_inside_promoted_with_type2 = bool(
            st.get('born_inside_outcome') == 'PROMOTED' and zone_flags.get('enter_b')
        )
        if (
            bool(zone_state and zone_state.inside_a)
            and not st.get('born_inside_pending')
            and not born_inside_promoted_with_type2
            and not st.get('lifecycle_handoff_pending')
            and 1 not in st['events']
            and 1 in self.allowed_events
            and can_type1
        ):
            self.emit_event(track_id, 1, frame_idx, frame, {'captureTime': timestamp}, st)
            st['events'].add(1)
        if type2_ready and 2 not in st['events'] and 2 in self.allowed_events:
            if 1 in self.allowed_events and 1 not in st['events']:
                self.emit_event(track_id, 1, frame_idx, frame, {
                    'captureTime': timestamp,
                    'sequenceBackfill': True,
                }, st)
                st['events'].add(1)
            self.emit_event(track_id, 2, frame_idx, frame, {'captureTime': timestamp}, st)
            st['events'].add(2)
        if (
            st.get('water_consecutive_frames', 0) >= self.water_confirm_frames
            and 3 not in st['events']
            and 3 in self.allowed_events
        ):
            st['water_detected'] = True
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
            self._finalize_vehicle_class(st, 'type4')
            self.emit_event(track_id, 4, frame_idx, frame, {
                'captureTime': timestamp,
                'washDuration': round(duration_val, 2),
            }, st)
            st['events'].add(4)
        st['washing'] = washing_now
        self._update_wheel_track_activity(track_id, st, frame_ts=time.time())
        if not st['washing']:
            st['washing_confirmed'] = False
        st['debug'] = {
            'state': 'washing' if st.get('washing') else 'idle',
            'wash_duration': round(st.get('wash_duration', 0.0), 1),
            'water': bool(washing_now),
            'water_consecutive_frames': int(st.get('water_consecutive_frames', 0) or 0),
            'plate': st.get('plate_text', ''),
            'zone_a': bool(zone_state and zone_state.inside_a),
            'zone_b': bool(zone_state and zone_state.inside_b),
            'zone_a_elapsed': zone_a_elapsed,
            'zone_b_elapsed': anchor_elapsed,
            'water_detected': bool(st.get('water_detected')),
            'zone_a_state': getattr(zone_state, 'zone_a_state', 'UNSEEN'),
            'zone_a_region': getattr(zone_state, 'zone_a_region', 'INVALID'),
            'zone_a_signed_distance': getattr(zone_state, 'signed_distance', None),
            'zone_a_dynamic_margin': getattr(zone_state, 'dynamic_margin', 0.0),
        }

        if (
            self.per_id_video_enabled
            and not st.get('type2_qualified')
            and st.get('record_start_frame') is not None
            and self.pre_type2_video_segment_seconds > 0.0
        ):
            segment_start = st.get('record_segment_start_frame')
            if segment_start is None:
                segment_start = st.get('record_start_frame', frame_idx)
                st['record_segment_start_frame'] = segment_start
            segment_elapsed = (frame_idx - int(segment_start)) / max(self.fps, 1e-6)
            if segment_elapsed >= self.pre_type2_video_segment_seconds:
                st['pre_type2_rotate_requested'] = True

        if (
            st.get('type2_qualified')
            and not st.get('closed')
            and 5 not in st.get('events', set())
            and self.post_type2_force_finalize_seconds > 0.0
        ):
            raw_type2_frame = st.get('type2_qualified_frame', frame_idx)
            try:
                type2_frame = int(raw_type2_frame)
            except (TypeError, ValueError):
                type2_frame = frame_idx
            if type2_frame < 0:
                type2_frame = frame_idx
            qualified_elapsed = (frame_idx - type2_frame) / max(self.fps, 1e-6)
            if qualified_elapsed >= self.post_type2_force_finalize_seconds:
                reasons = st.setdefault('abnormal_reasons', set())
                if not isinstance(reasons, set):
                    reasons = set(reasons)
                    st['abnormal_reasons'] = reasons
                reasons.add('OVER_15_MINUTES_AFTER_TYPE2')
                self._complete_type2_lifecycle(
                    track_id,
                    st,
                    frame_idx,
                    frame,
                    type4_backfill_reason='type2_dwell_timeout_15m',
                    completion_reason='type2_dwell_timeout_15m',
                    force_video_stop=True,
                    trace_frame_idx=frame_idx,
                )

    def flush_inactive(self, active_ids, frame_idx, on_track_timeout=None, timeout_reason='track_lost',
                       capture_ts=None):
        active_ids = active_ids or set()
        to_remove = []
        for tid, st in self.tracks.items():
            if tid in active_ids:
                continue
            if st.get('_lifecycle_superseded'):
                to_remove.append(tid)
                continue
            lifecycle = self.lifecycle_manager.get(tid)
            now_capture_ts = capture_ts if capture_ts is not None else self._capture_timestamp(frame_idx)
            if lifecycle is not None and now_capture_ts is not None:
                if lifecycle.active_tracker_id == tid:
                    self.lifecycle_manager.mark_lost(tid, capture_ts=now_capture_ts)
                    self.trace_record('lifecycle_lost', {
                        'frameIdx': int(frame_idx),
                        'trackId': int(tid),
                        'eventId': lifecycle.event_id,
                        'captureTs': float(now_capture_ts),
                    })
                if self.lifecycle_manager.is_waiting(tid, now_capture_ts):
                    continue
            timed_out = (
                frame_idx - st.get('last_frame_idx', frame_idx) >= self.timeout_frames
                if now_capture_ts is None or lifecycle is None
                else True
            )
            if timed_out:
                born_inside_rejected = bool(
                    st.get('born_inside_pending')
                    and str(st.get('born_inside_outcome', '') or '') == 'CANDIDATE'
                )
                if born_inside_rejected:
                    st['born_inside_pending'] = False
                    st['born_inside_outcome'] = 'REJECTED'
                    st['candidate_rejection_reason'] = 'TRACK_LOST_IN_ZONE_A_TIMEOUT'
                    self.trace_record('born_inside', {
                        'frameIdx': int(frame_idx),
                        'trackId': int(tid),
                        'action': 'rejected',
                        'reason': 'track_lost_in_zone_a_timeout',
                    })
                lost_inside_a_timeout = bool(
                    not born_inside_rejected
                    and st.get('type2_qualified')
                    and st.get('zone_a_seen')
                    and not st.get('zone_a_exited')
                    and timeout_reason == 'track_lost'
                )
                if lost_inside_a_timeout:
                    reasons = st.setdefault('abnormal_reasons', set())
                    if not isinstance(reasons, set):
                        reasons = set(reasons)
                        st['abnormal_reasons'] = reasons
                    reasons.add('TRACK_LOST_IN_ZONE_A_TIMEOUT')
                    self.trace_record('lifecycle_timeout', {
                        'frameIdx': int(frame_idx),
                        'trackId': int(tid),
                        'reason': 'track_lost_in_zone_a_timeout',
                        'type2Qualified': True,
                    })
                    if st.get('record_start_frame') is not None and st.get('record_stop_frame') is None:
                        st['record_stop_frame'] = frame_idx
                can_type5 = self._can_emit_type5(st)
                if can_type5:
                    self._complete_type2_lifecycle(
                        tid,
                        st,
                        st.get('last_frame_idx', frame_idx),
                        st.get('last_frame'),
                        type4_backfill_reason='type5_anchor_missing_timeout',
                        completion_reason='anchor_missing_timeout',
                        force_video_stop=False,
                        trace_frame_idx=frame_idx,
                    )
                if st.get('record_start_frame') is not None and st.get('record_stop_frame') is None:
                    extra_frames = self._record_tail_frames_for_event(5, st)
                    last_idx = st.get('last_frame_idx', frame_idx)
                    st['record_stop_frame'] = last_idx + extra_frames
                if self.single_lifecycle_events and (5 in st['events'] or born_inside_rejected):
                    st['closed'] = True
                if lifecycle is not None:
                    self.lifecycle_manager.finalize(tid)

                stop_f = st.get('record_stop_frame')
                tail_pending = stop_f is not None and frame_idx < stop_f

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
            lifecycle = self.lifecycle_manager.get(tid)
            key = lifecycle.event_id if lifecycle is not None else f'{self.camera_id}_{tid}'
            self.upload_buffer.pop(key, None)
            self.tracks.pop(tid, None)

    def emit_event(self, track_id, event_type, frame_idx, frame, payload, track_state):
        if event_type not in self.allowed_events:
            return
        vehicle_type = payload.get('vehicleType') or self._resolve_vehicle_type(track_state)
        self._emit_event_core(track_id, event_type, frame_idx, frame, payload, track_state, vehicle_type)

    def _record_event_sequence_issue(self, track_id, event_type, frame_idx, track_state, previous_stage):
        issue = {
            'trackId': int(track_id),
            'eventType': int(event_type),
            'previousStage': int(previous_stage),
            'frameIdx': int(frame_idx),
            'reason': 'event_stage_regression',
        }
        issues = track_state.setdefault('event_sequence_issues', [])
        issues.append(issue)
        if len(issues) > 32:
            del issues[:-32]
        reasons = track_state.setdefault('abnormal_reasons', set())
        if isinstance(reasons, set):
            reasons.add(f'OUT_OF_ORDER_TYPE{event_type}_AFTER_TYPE{previous_stage}')
        print(
            f'[event-sequence] suppress regression track={track_id} '
            f'type={event_type} previous={previous_stage} frame={frame_idx}'
        )
        self.trace_record('sequence_issue', issue)

    def _event_stage_allowed(self, track_id, event_type, frame_idx, track_state):
        try:
            stage = int(event_type)
        except (TypeError, ValueError):
            return False
        lifecycle = self.lifecycle_manager.get(track_id)
        lifecycle_stages = lifecycle.stages if lifecycle is not None else set()
        if stage in lifecycle_stages:
            return False
        previous_stage = max(
            int(track_state.get('event_stage_max', 0) or 0),
            max(lifecycle_stages) if lifecycle_stages else 0,
        )
        if stage < previous_stage:
            self._record_event_sequence_issue(
                track_id,
                stage,
                frame_idx,
                track_state,
                previous_stage,
            )
            return False
        if stage > previous_stage:
            track_state['event_stage_max'] = stage
        return True

    def _emit_event_core(self, track_id, event_type, frame_idx, frame, payload, track_state, vehicle_type):
        if not self._event_stage_allowed(track_id, event_type, frame_idx, track_state):
            return False
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
        capture_ts = self._capture_timestamp(frame_idx) or time.time()
        lifecycle = self.lifecycle_manager.get(track_id)
        if lifecycle is None:
            lifecycle = self.lifecycle_manager.create(track_id, vehicle_type, capture_ts=capture_ts)
        locked_plate = normalize_plate_candidate_text(track_state.get('plate_text_locked') or '')
        locked_plate_valid = bool(locked_plate and is_valid_plate(locked_plate))
        self.lifecycle_manager.touch(
            track_id,
            capture_ts=capture_ts,
            vehicle_class=vehicle_type,
            plate_text=locked_plate if locked_plate_valid else '',
            plate_box=track_state.get('last_plate_box') if locked_plate_valid else None,
            plate_edge=self._lifecycle_zone_edge(
                track_state.get('last_anchor'),
                track_state.get('last_plate_box'),
            ),
            motion_direction=str(track_state.get('anchor_motion_direction', 'unknown') or 'unknown'),
            plate_color=track_state.get('plate_color_locked', '') if locked_plate_valid else '',
            plate_color_conf=track_state.get('plate_color_locked_conf', 0.0) if locked_plate_valid else 0.0,
            plate_type=track_state.get('plate_type', '') if locked_plate_valid else '',
        )
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
        session_id = lifecycle.event_id
        track_state['session_id'] = session_id
        if event_type == 1:
            prev_start = track_state.get('record_start_frame')
            if prev_start is None or frame_idx < prev_start:
                track_state['record_start_frame'] = frame_idx
            if track_state.get('record_first_start_frame') is None:
                track_state['record_first_start_frame'] = frame_idx
            if track_state.get('record_segment_start_frame') is None:
                track_state['record_segment_start_frame'] = frame_idx
        if track_state.get('record_start_frame') is None:
            track_state['record_start_frame'] = frame_idx
        if event_type == 5:
            self._wait_for_wheel_results_for_type5(track_id, track_state)
            extra_frames = (
                0
                if payload.get('forceVideoStop')
                else self._record_tail_frames_for_event(event_type, track_state)
            )
            stop_frame = frame_idx + extra_frames
            prev_stop = track_state.get('record_stop_frame')
            if prev_stop is None or stop_frame > prev_stop:
                track_state['record_stop_frame'] = stop_frame
            self._enqueue_wheel_photos(track_state, track_id=track_id, force=True)
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
        if payload.get('sequenceBackfill'):
            event['sequenceBackfill'] = True
            event['backfillReason'] = str(payload.get('backfillReason') or '')
        if payload.get('forcedCompletionReason'):
            event['forcedCompletionReason'] = str(payload.get('forcedCompletionReason') or '')
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
            event['directionSource'] = track_state.get('direction_source', 'unknown')
        plate_color, plate_color_conf = self._infer_plate_color(track_state)
        event['plateColor'] = plate_color
        event['plateColorConfidence'] = plate_color_conf
        event['plateColorSource'] = track_state.get('plate_color_source', 'unknown')
        if plate_color == '黄绿色':
            locked_plate_text = normalize_plate_candidate_text(track_state.get('plate_text_locked', ''))
            color_state = (
                (track_state.get('plate_color_evidence_by_text') or {}).get(locked_plate_text)
                if locked_plate_text else None
            )
            fusion_summary = (color_state or {}).get('fusion_summary')
            if isinstance(fusion_summary, dict):
                event['plateColorEvidence'] = dict(fusion_summary)
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
        if event_type == 6:
            event['perIdVideoEnabled'] = bool(payload.get('perIdVideoEnabled', False))
        self.lifecycle_manager.mark_stage(track_id, event_type)
        if event_type >= 5:
            self.lifecycle_manager.finalize(track_id)
        if track_state.get('wash_start_time') and not event.get('washStartTime'):
            event['washStartTime'] = track_state.get('wash_start_time')
        self.trace_record('event', {
            'frameIdx': int(frame_idx),
            'trackId': int(track_id),
            'event': event,
        })
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
        event_path = None
        if self.enable_event_disk:
            candidate_event_path = self.events_dir / f'{self.camera_id}_{track_id}_t{event_type}_{frame_idx}.json'
            try:
                with candidate_event_path.open('w', encoding='utf-8') as f:
                    json.dump(event, f, ensure_ascii=False, indent=2)
                event_path = candidate_event_path
            except Exception:
                pass
        try:
            self._record_event_evidence(event, track_state, frame_idx, event_path=event_path)
        except Exception as exc:
            for line in self.log_throttler.record(
                key='evidence.index.write',
                message=f'[evidence] failed to update event index id={event.get("id", "")}: {exc}',
                window_seconds=30.0,
            ):
                print(line)
        t_json = time.perf_counter()
        print(f"[EVENT] cam={self.camera_id} track={track_id} type={event_type} time={event['captureTime']}")
        if self.event_log_path:
            try:
                with self.event_log_path.open('a', encoding='utf-8') as f:
                    f.write(f"{self.camera_id},{track_id},{event_type},{event['captureTime']},{frame_idx},"
                            f"0,{round(track_state.get('wash_duration',0.0),2)},"
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
        return True

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

    @staticmethod
    def _reset_plate_switch_streak(track_state):
        track_state['plate_text_switch_candidate'] = ''
        track_state['plate_text_switch_streak'] = 0

    def _update_locked_plate_text(
        self,
        track_id,
        track_state,
        frame_idx,
        observed_text='',
        observed_trusted=False,
        observed_switch_trusted=False,
    ):
        frame_idx = int(frame_idx)
        observed_text = normalize_plate_candidate_text(observed_text)
        if not observed_trusted or not is_valid_plate(observed_text):
            return

        locked_text = normalize_plate_candidate_text(track_state.get('plate_text_locked', ''))
        if not locked_text:
            previous_text = normalize_plate_candidate_text(track_state.get('plate_initial_candidate', ''))
            if observed_text == previous_text:
                track_state['plate_initial_streak'] = int(track_state.get('plate_initial_streak', 0) or 0) + 1
            else:
                track_state['plate_initial_candidate'] = observed_text
                track_state['plate_initial_streak'] = 1
            if int(track_state.get('plate_initial_streak', 0) or 0) >= self._event_plate_lock_required_hits(track_state):
                track_state['plate_text_locked'] = observed_text
                track_state['plate_text_locked_is_guess'] = False
                track_state['plate_initial_candidate'] = ''
                track_state['plate_initial_streak'] = 0
                self._reset_plate_switch_streak(track_state)
                self._activate_plate_color_for_text(track_state, observed_text)
                self.trace_record('plate_text_lock', {
                    'frameIdx': frame_idx,
                    'trackId': int(track_id),
                    'text': observed_text,
                    'requiredHits': self._event_plate_lock_required_hits(track_state),
                })
            return

        if observed_text == locked_text:
            self._reset_plate_switch_streak(track_state)
            return
        if not observed_switch_trusted:
            self._reset_plate_switch_streak(track_state)
            return

        previous_text = normalize_plate_candidate_text(track_state.get('plate_text_switch_candidate', ''))
        if observed_text == previous_text:
            track_state['plate_text_switch_streak'] = int(track_state.get('plate_text_switch_streak', 0) or 0) + 1
        else:
            track_state['plate_text_switch_candidate'] = observed_text
            track_state['plate_text_switch_streak'] = 1
        if int(track_state.get('plate_text_switch_streak', 0) or 0) < self.plate_correction_confirm_hits:
            return

        previous_locked_text = locked_text
        track_state['plate_text_locked'] = observed_text
        track_state['plate_text_locked_is_guess'] = False
        self._reset_plate_switch_streak(track_state)
        self._activate_plate_color_for_text(track_state, observed_text)
        self.trace_record('plate_text_switch', {
            'frameIdx': frame_idx,
            'trackId': int(track_id),
            'lockedText': previous_locked_text,
            'candidateText': observed_text,
            'action': 'switched',
            'reason': 'continuous_trusted_evidence',
            'requiredHits': self.plate_correction_confirm_hits,
        })

    def _event_plate_lock_required_hits(self, track_state):
        del track_state
        return self.event_plate_lock_frames

    @staticmethod
    def _normalize_plate_color(color):
        value = str(color or '').strip()
        aliases = {
            'blue': '蓝色',
            'yellow': '黄色',
            'green': '绿色',
            'yellow_green': '黄绿色',
            'yellow-green': '黄绿色',
            '黄绿': '黄绿色',
            '黄绿牌': '黄绿色',
            'white': '白色',
            'black': '黑色',
        }
        return aliases.get(value.lower(), value)

    def _record_yellow_green_fusion_evidence(
        self,
        track_state,
        plate_text,
        color,
        color_conf,
        frame_idx,
    ):
        if not self.plate_yellow_green_fusion_enabled:
            return None
        if color not in {'黄色', '绿色', '蓝色', '白色', '黑色'}:
            return None
        confidence = float(color_conf or 0.0)
        if confidence < self.plate_yellow_green_min_confidence:
            return None

        buckets = track_state.get('plate_color_fusion_evidence_by_text')
        if not isinstance(buckets, dict):
            buckets = {}
            track_state['plate_color_fusion_evidence_by_text'] = buckets
        history = buckets.get(plate_text)
        if not isinstance(history, list):
            history = []
        frame_idx = int(frame_idx)
        cutoff = frame_idx - self.plate_yellow_green_window_frames + 1
        history = [
            item for item in history
            if int(item.get('frame', -1)) >= cutoff and int(item.get('frame', -1)) != frame_idx
        ]
        history.append({
            'frame': frame_idx,
            'color': color,
            'confidence': confidence,
        })
        buckets[plate_text] = history

        yellow = [item for item in history if item['color'] == '黄色']
        green = [item for item in history if item['color'] == '绿色']
        conflicts = [item for item in history if item['color'] not in {'黄色', '绿色'}]
        yellow_green_total = len(yellow) + len(green)
        evidence_total = yellow_green_total + len(conflicts)
        required_per_color = self.plate_yellow_green_min_hits_per_color
        required_total = max(5, required_per_color * 2 + 1)
        minority_ratio = (
            min(len(yellow), len(green)) / yellow_green_total
            if yellow_green_total > 0 else 0.0
        )
        conflict_ratio = len(conflicts) / evidence_total if evidence_total > 0 else 0.0
        confirmed = bool(
            len(yellow) >= required_per_color
            and len(green) >= required_per_color
            and yellow_green_total >= required_total
            and minority_ratio >= 0.25
            and len(conflicts) <= 1
            and conflict_ratio <= 0.20
        )
        summary = {
            'windowFrames': self.plate_yellow_green_window_frames,
            'yellowHits': len(yellow),
            'greenHits': len(green),
            'conflictHits': len(conflicts),
            'yellowMeanConfidence': round(
                sum(item['confidence'] for item in yellow) / len(yellow), 6
            ) if yellow else 0.0,
            'greenMeanConfidence': round(
                sum(item['confidence'] for item in green) / len(green), 6
            ) if green else 0.0,
            'minorityRatio': round(minority_ratio, 6),
            'conflictRatio': round(conflict_ratio, 6),
            'confirmed': confirmed,
        }
        return summary

    def _activate_plate_color_for_text(self, track_state, plate_text):
        plate_text = normalize_plate_candidate_text(plate_text)
        buckets = track_state.get('plate_color_evidence_by_text') or {}
        state = buckets.get(plate_text) if isinstance(buckets, dict) else None
        stable_color = self._normalize_plate_color((state or {}).get('stable_color', ''))
        if stable_color not in self.allowed_plate_colors:
            stable_color = ''
        track_state['plate_color_locked'] = stable_color
        track_state['plate_color_locked_conf'] = float((state or {}).get('stable_conf', 0.0) or 0.0)
        track_state['plate_color_locked_text'] = plate_text if stable_color else ''
        track_state['plate_color_latest'] = stable_color
        track_state['plate_color_latest_conf'] = track_state['plate_color_locked_conf'] if stable_color else 0.0
        votes_by_text = track_state.get('plate_color_votes_by_text') or {}
        track_state['plate_color_votes'] = dict(votes_by_text.get(plate_text) or {})

    def _update_locked_plate_color(
        self,
        track_id,
        track_state,
        plate_text,
        color,
        color_conf,
        frame_idx,
        trusted=False,
    ):
        color = self._normalize_plate_color(color)
        plate_text = normalize_plate_candidate_text(plate_text)
        locked_text = normalize_plate_candidate_text(track_state.get('plate_text_locked', ''))
        trace = {
            'frameIdx': int(frame_idx),
            'trackId': int(track_id),
            'text': plate_text,
            'color': color,
            'confidence': float(color_conf or 0.0),
        }
        if not color or not plate_text or not is_valid_plate(plate_text):
            trace.update(action='rejected', reason='missing_or_invalid_text')
            self.trace_record('plate_color_evidence', trace)
            return
        if not trusted:
            trace.update(action='rejected', reason='untrusted_text_evidence')
            self.trace_record('plate_color_evidence', trace)
            return
        frame_idx = int(frame_idx)
        fusion_summary = self._record_yellow_green_fusion_evidence(
            track_state,
            plate_text,
            color,
            color_conf,
            frame_idx,
        )
        if fusion_summary is not None:
            trace['yellowGreenFusion'] = fusion_summary
        if fusion_summary and fusion_summary.get('confirmed'):
            buckets = track_state.get('plate_color_evidence_by_text')
            if not isinstance(buckets, dict):
                buckets = {}
                track_state['plate_color_evidence_by_text'] = buckets
            state = buckets.get(plate_text)
            if not isinstance(state, dict):
                state = {}
                buckets[plate_text] = state
            state['stable_color'] = '黄绿色'
            state['stable_conf'] = min(
                float(fusion_summary.get('yellowMeanConfidence', 0.0) or 0.0),
                float(fusion_summary.get('greenMeanConfidence', 0.0) or 0.0),
            )
            state['candidate_color'] = ''
            state['candidate_hits'] = 0
            state['candidate_conf_sum'] = 0.0
            state['fusion_summary'] = dict(fusion_summary)
            if locked_text == plate_text:
                self._activate_plate_color_for_text(track_state, plate_text)
            trace.update(action='locked', reason='mixed_yellow_green_evidence')
            self.trace_record('plate_color_evidence', trace)
            return

        if color_conf < self.plate_color_min_confidence:
            trace.update(action='rejected', reason='low_color_confidence')
            self.trace_record('plate_color_evidence', trace)
            return
        if color not in self.allowed_plate_colors:
            trace.update(action='rejected', reason='business_color_not_allowed')
            self.trace_record('plate_color_evidence', trace)
            return

        buckets = track_state.get('plate_color_evidence_by_text')
        if not isinstance(buckets, dict):
            buckets = {}
            track_state['plate_color_evidence_by_text'] = buckets
        state = buckets.get(plate_text)
        if not isinstance(state, dict):
            state = {
                'stable_color': '',
                'stable_conf': 0.0,
                'candidate_color': '',
                'candidate_hits': 0,
                'candidate_conf_sum': 0.0,
            }
            buckets[plate_text] = state

        votes_by_text = track_state.get('plate_color_votes_by_text')
        if not isinstance(votes_by_text, dict):
            votes_by_text = {}
            track_state['plate_color_votes_by_text'] = votes_by_text
        votes = votes_by_text.setdefault(plate_text, {})
        vote = votes.setdefault(color, {'hits': 0, 'sum_conf': 0.0})
        vote['hits'] = int(vote.get('hits', 0) or 0) + 1
        vote['sum_conf'] = float(vote.get('sum_conf', 0.0) or 0.0) + float(color_conf)
        trace['colorVotes'] = votes

        stable_color = self._normalize_plate_color(state.get('stable_color', ''))
        if stable_color == '黄绿色':
            state['candidate_color'] = ''
            state['candidate_hits'] = 0
            state['candidate_conf_sum'] = 0.0
            action = 'kept_fused'
        elif color == stable_color:
            state['stable_conf'] = max(float(state.get('stable_conf', 0.0) or 0.0), float(color_conf))
            state['candidate_color'] = ''
            state['candidate_hits'] = 0
            state['candidate_conf_sum'] = 0.0
            action = 'kept'
        else:
            if state.get('candidate_color') == color:
                state['candidate_hits'] = int(state.get('candidate_hits', 0) or 0) + 1
                state['candidate_conf_sum'] = float(state.get('candidate_conf_sum', 0.0) or 0.0) + float(color_conf)
            else:
                state['candidate_color'] = color
                state['candidate_hits'] = 1
                state['candidate_conf_sum'] = float(color_conf)
            required_hits = self.plate_color_correction_hits if stable_color else self.plate_color_lock_frames
            action = 'buffered'
            if int(state.get('candidate_hits', 0) or 0) >= required_hits:
                hits = max(1, int(state.get('candidate_hits', 0) or 0))
                state['stable_color'] = color
                state['stable_conf'] = float(state.get('candidate_conf_sum', 0.0) or 0.0) / hits
                state['candidate_color'] = ''
                state['candidate_hits'] = 0
                state['candidate_conf_sum'] = 0.0
                action = 'switched' if stable_color else 'locked'

        if locked_text == plate_text:
            self._activate_plate_color_for_text(track_state, plate_text)
        else:
            action = 'buffered_for_text'
        trace['action'] = action
        self.trace_record('plate_color_evidence', trace)

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

        locked_color = self._normalize_plate_color(track_state.get('plate_color_locked', ''))
        locked_color_text = normalize_plate_candidate_text(track_state.get('plate_color_locked_text', ''))
        if locked_color not in self.allowed_plate_colors or locked_color_text != normalize_plate_candidate_text(locked_text):
            locked_color = ''
        latest_color = self._normalize_plate_color(track_state.get('plate_color_latest', ''))
        if latest_color not in self.allowed_plate_colors:
            latest_color = ''
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
        if not bool(getattr(self, 'per_id_video_enabled', False)):
            return 0
        return int(max(self.fps, 1.0) * self.per_id_video_tail_seconds)

    def _record_tail_frames_for_event(self, event_type, track_state=None):
        if (
            int(event_type) == 5
            and isinstance(track_state, dict)
            and 'TRACK_LOST_IN_ZONE_A_TIMEOUT' in (track_state.get('abnormal_reasons') or set())
        ):
            return int(round(max(self.fps, 1.0) * self.track_lost_grace_seconds))
        return self._record_tail_frames()

    def _can_emit_type1(self, track_state):
        if self.min_type1_track_frames <= 1:
            return True
        stable_zone_a_frames = int(track_state.get('zone_a_dwell_frames', 0) or 0) + 1
        return stable_zone_a_frames >= self.min_type1_track_frames

    def _can_emit_type5(self, track_state):
        return bool(track_state.get('type2_qualified'))

    def _complete_type2_lifecycle(
        self,
        track_id,
        track_state,
        frame_idx,
        frame,
        type4_backfill_reason,
        completion_reason,
        force_video_stop=False,
        trace_frame_idx=None,
    ):
        if (
            not self._can_emit_type5(track_state)
            or 5 in track_state.get('events', set())
            or 5 not in self.allowed_events
        ):
            return False
        timestamp = self.frame_timestamp(frame_idx)
        track_state['wash_end_time'] = track_state.get('wash_end_time') or timestamp
        duration_val = self._compute_effective_wash_duration(track_state, frame_idx)
        track_state['wash_duration'] = duration_val
        if 4 not in track_state['events'] and 4 in self.allowed_events:
            self._finalize_vehicle_class(track_state, 'type4_backfill')
            self.emit_event(track_id, 4, frame_idx, frame, {
                'captureTime': timestamp,
                'washDuration': round(duration_val, 2),
                'sequenceBackfill': True,
                'backfillReason': str(type4_backfill_reason or ''),
            }, track_state)
            track_state['events'].add(4)
            self.trace_record('type4_gate', {
                'frameIdx': int(trace_frame_idx if trace_frame_idx is not None else frame_idx),
                'trackId': int(track_id),
                'action': 'backfill',
                'reason': str(type4_backfill_reason or ''),
            })
        self._finalize_vehicle_class(track_state, 'type5')
        self._mark_type5_abnormal_reasons(track_state)
        payload = {
            'captureTime': timestamp,
            'washDuration': round(duration_val, 2),
        }
        if force_video_stop:
            payload['forceVideoStop'] = True
        if completion_reason:
            payload['forcedCompletionReason'] = str(completion_reason)
        self.emit_event(track_id, 5, frame_idx, frame, payload, track_state)
        track_state['events'].add(5)
        if self.single_lifecycle_events:
            track_state['closed'] = True
        self.trace_record('type5_gate', {
            'frameIdx': int(trace_frame_idx if trace_frame_idx is not None else frame_idx),
            'trackId': int(track_id),
            'action': 'release',
            'reason': str(completion_reason or ''),
            'zoneAExited': bool(track_state.get('zone_a_exited')),
            'abnormal': bool(track_state.get('abnormal_reasons')),
            'forceVideoStop': bool(force_video_stop),
        })
        return True

    def _mark_type5_abnormal_reasons(self, track_state):
        reasons = track_state.get('abnormal_reasons')
        if reasons is None:
            reasons = set()
            track_state['abnormal_reasons'] = reasons
        if 2 not in track_state.get('events', set()):
            reasons.add('MISSING_TYPE2')
        if track_state.get('type2_qualified') and 4 not in track_state.get('events', set()):
            reasons.add('MISSING_TYPE4')
        return reasons

    def _avg(self, values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _select_vehicle_class(track_state):
        counts = dict(track_state.get('class_counts') or {})
        stage_counts = dict(track_state.get('wash_stage_class_counts') or {})
        if not counts and not stage_counts:
            return str(track_state.get('last_vehicle_label') or '')
        scores = {label: int(count or 0) for label, count in counts.items() if label}
        for label, count in stage_counts.items():
            if label:
                scores[label] = scores.get(label, 0) + 2 * int(count or 0)
        if not scores:
            return ''
        return max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]

    def _start_wash_stage_vehicle_class(self, track_state, vehicle_label=''):
        if not track_state.get('vehicle_cls_at_type2'):
            track_state['vehicle_cls_at_type2'] = (
                self._select_vehicle_class(track_state)
                or str(vehicle_label or '')
            )
        label = str(vehicle_label or '').strip()
        if label:
            stage_counts = track_state.get('wash_stage_class_counts') or {}
            stage_counts[label] = stage_counts.get(label, 0) + 1
            track_state['wash_stage_class_counts'] = stage_counts
        track_state['vehicle_cls'] = self._select_vehicle_class(track_state)

    def _finalize_vehicle_class(self, track_state, reason):
        if track_state.get('vehicle_cls_frozen') and track_state.get('vehicle_cls_locked'):
            return track_state['vehicle_cls_locked']
        selected = (
            self._select_vehicle_class(track_state)
            or str(track_state.get('vehicle_cls_at_type2') or '')
            or str(track_state.get('vehicle_cls') or '')
            or str(track_state.get('last_vehicle_label') or '')
        )
        if selected:
            track_state['vehicle_cls'] = selected
            track_state['vehicle_cls_locked'] = selected
            track_state['vehicle_cls_frozen'] = True
            track_state['vehicle_cls_lock_reason'] = str(reason or 'lifecycle_finalized')
        return selected

    def _infer_plate_color(self, track_state):
        locked_text = normalize_plate_candidate_text(track_state.get('plate_text_locked', ''))
        locked_color_text = normalize_plate_candidate_text(track_state.get('plate_color_locked_text', ''))
        locked_color = self._normalize_plate_color(track_state.get('plate_color_locked', ''))
        if locked_color not in self.allowed_plate_colors or locked_color_text != locked_text:
            locked_color = ''
        if locked_color:
            try:
                locked_conf = float(track_state.get('plate_color_locked_conf', 0.0) or 0.0)
            except (TypeError, ValueError):
                locked_conf = 0.0
            track_state['plate_color_source'] = (
                'mixed_yellow_green_evidence'
                if locked_color == '黄绿色'
                else 'locked_text_evidence'
            )
            return locked_color, locked_conf
        latest_color = self._normalize_plate_color(track_state.get('plate_color_latest', ''))
        if latest_color not in self.allowed_plate_colors:
            latest_color = ''
        if latest_color:
            try:
                latest_conf = float(track_state.get('plate_color_latest_conf', 0.0) or 0.0)
            except (TypeError, ValueError):
                latest_conf = 0.0
            track_state['plate_color_source'] = 'latest_text_evidence'
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
                track_state['plate_color_source'] = 'vehicle_type_fallback'
                return '绿色', 0.0
            track_state['plate_color_source'] = 'vehicle_type_fallback'
            return '蓝色', 0.0
        if vehicle == 'blue truck':
            track_state['plate_color_source'] = 'vehicle_type_fallback'
            return '蓝色', 0.0
        if vehicle in ('yellow truck', 'dump truck'):
            track_state['plate_color_source'] = 'vehicle_type_fallback'
            return '黄色', 0.0
        if vehicle == 'wuxiao':
            track_state['plate_color_source'] = 'vehicle_type_fallback'
            return '蓝色', 0.0
        default_plate_color = self._normalize_plate_color(self.default_plate_color)
        if default_plate_color in self.allowed_plate_colors:
            track_state['plate_color_source'] = 'configured_default'
            return default_plate_color, self.default_plate_color_conf
        track_state['plate_color_source'] = 'unknown'
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
        color = self._normalize_plate_color(st.get('plate_color_locked', ''))
        color_text = normalize_plate_candidate_text(st.get('plate_color_locked_text', ''))
        if color not in self.allowed_plate_colors or color_text != text:
            color = ''
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
            'water': dbg.get('water', False),
            'water_consecutive_frames': dbg.get('water_consecutive_frames', 0),
            'wash_duration': dbg.get('wash_duration', 0.0),
            'plate': dbg.get('plate', ''),
            'zone_a': dbg.get('zone_a', False),
            'zone_b': dbg.get('zone_b', False),
            'zone_b_elapsed': dbg.get('zone_b_elapsed', 0),
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
        metrics['upload_buffered'] = sum(len(v) for v in self.upload_buffer.values())
        metrics['capture_base64_cached'] = len(self._capture_base64_cache)
        metrics['frame_timing_entries'] = len(self.frame_timing)
        metrics['frame_timing_limit'] = self.frame_timing_max_entries
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
                'perIdVideoEnabled': bool(event.get('perIdVideoEnabled', False)),
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
            f"class={candidate.get('className', '')} score={float(candidate.get('score', 0.0) or 0.0):.3f} "
            f"center={self._wheel_candidate_center_distance(candidate):.1f}"
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

    @staticmethod
    def _wheel_candidate_center_distance(item, default=float('inf')):
        if not isinstance(item, dict) or 'centerDistance' not in item:
            return float(default)
        try:
            value = float(item.get('centerDistance'))
        except (TypeError, ValueError):
            return float(default)
        if value != value or value < 0.0:
            return float(default)
        return value

    @staticmethod
    def _wheel_result_selection_key(item, ref_ts):
        if not isinstance(item, dict):
            return (float('inf'), 0.0, float('inf'), 0.0)
        center_distance = EventManager._wheel_candidate_center_distance(item)
        try:
            score = float(item.get('score', 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        try:
            capture_ts = float(item.get('capture_ts', ref_ts) or ref_ts)
        except (TypeError, ValueError):
            capture_ts = float(ref_ts)
        time_delta = abs(float(ref_ts) - capture_ts)
        return (center_distance, -score, time_delta, -capture_ts)

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
                'centerDistance': self._wheel_candidate_center_distance(item),
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
            current_key = self._wheel_result_selection_key(current, ref_ts)
            candidate_key = self._wheel_result_selection_key(candidate, ref_ts)
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
        self._enqueue_wheel_photos(track_state, now_ts=ref_ts, track_id=track_id, force=False)

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
            'centerDistance': self._wheel_candidate_center_distance(item),
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
            'centerDistance': self._wheel_candidate_center_distance(candidate),
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
        return min(candidates, key=EventManager._wheel_candidate_center_distance)

    @staticmethod
    def _wheel_photo_uploaded_urls(track_state):
        uploaded = track_state.setdefault('wheel_photo_uploaded_urls', set())
        if isinstance(uploaded, set):
            return uploaded
        if isinstance(uploaded, (list, tuple)):
            uploaded = {str(item) for item in uploaded if str(item)}
        else:
            uploaded = set()
        track_state['wheel_photo_uploaded_urls'] = uploaded
        return uploaded

    def _wheel_photo_bucket_ready(self, bucket_key, rep, now_ts):
        try:
            bucket_end = (int(bucket_key) + 1) * self.wheel_photo_bucket_seconds
        except (TypeError, ValueError):
            try:
                bucket_end = float(rep.get('capture_ts', 0.0) or 0.0) + self.wheel_photo_bucket_seconds
            except (TypeError, ValueError):
                return False
        return float(now_ts) >= bucket_end

    def _collect_wheel_photo_entries(self, track_state, now_ts=None, force=True, track_id=None):
        if not isinstance(track_state, dict):
            return []
        if not self.wheel_photo_uploader:
            return []
        now_ref = time.time() if now_ts is None else float(now_ts)
        history = track_state.get('wheel_photo_history')
        photos = []
        sides_with_history_photos = set()
        if isinstance(history, dict):
            for side in ('left', 'right'):
                side_history = history.get(side)
                if not isinstance(side_history, dict):
                    continue
                for bucket_key, bucket in side_history.items():
                    if not isinstance(bucket, dict):
                        continue
                    rep = bucket.get('representative')
                    if not isinstance(rep, dict):
                        continue
                    if rep.get('duplicatePhoto'):
                        continue
                    if not force and not self._wheel_photo_bucket_ready(bucket_key, rep, now_ref):
                        continue
                    sides_with_history_photos.add(side)
                    photos.append((float(rep.get('capture_ts', 0.0) or 0.0), rep))
        if force:
            locked = track_state.get('wheel_results_locked')
            if isinstance(locked, dict):
                for side in ('left', 'right'):
                    if side in sides_with_history_photos:
                        continue
                    entry = locked.get(side)
                    if not isinstance(entry, dict):
                        continue
                    photo_url = self._ensure_locked_wheel_photo_url(
                        side=side,
                        entry=entry,
                        track_state=track_state,
                        track_id=track_id,
                    )
                    class_name = str(entry.get('className') or '').strip()
                    clean_value = WHEEL_CLASS_NAME_TO_CLEAN_VALUE.get(class_name, 0)
                    if not photo_url or not clean_value:
                        continue
                    photos.append((
                        float(entry.get('capture_ts', 0.0) or 0.0),
                        {
                            'photoUrl': photo_url,
                            'type': WHEEL_SIDE_TO_PHOTO_TYPE.get(side, ''),
                            'cleanValue': clean_value,
                            'capture_ts': float(entry.get('capture_ts', 0.0) or 0.0),
                            '_sourceEntry': entry,
                        },
                    ))
        photos.sort(key=lambda item: item[0])
        return photos

    def _enqueue_wheel_photos(self, track_state, now_ts=None, force=True, track_id=None):
        if not self.wheel_photo_uploader or not isinstance(track_state, dict):
            return
        uploaded_urls = self._wheel_photo_uploaded_urls(track_state)
        photos = self._collect_wheel_photo_entries(
            track_state,
            now_ts=now_ts,
            force=force,
            track_id=track_id,
        )
        for _, entry in photos:
            photo_url = self._absolute_wheel_photo_url(entry.get('photoUrl'))
            if not photo_url or photo_url in uploaded_urls:
                continue
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
                uploaded_urls.add(photo_url)
                entry['wheelPhotoUploaded'] = True
                entry['wheelPhotoUploadedAt'] = time.time()
                source_entry = entry.get('_sourceEntry')
                if isinstance(source_entry, dict):
                    source_entry['wheelPhotoUploaded'] = True
                    source_entry['wheelPhotoUploadedAt'] = entry['wheelPhotoUploadedAt']
            except Exception as exc:
                print(f'[wheel-photo] failed to enqueue ({photo_url}): {exc}')

    def _resolve_direction(self, track_state):
        state = None
        if track_state:
            state = track_state.get('zone_state')
        direction_code, direction_label = self.zone_mgr.resolve_direction(state)
        if direction_code:
            if track_state is not None:
                track_state['direction_source'] = 'zone_geometry'
            return direction_code, direction_label
        reasons = track_state.get('abnormal_reasons', set()) if track_state else set()
        if not isinstance(reasons, set):
            reasons = set(reasons or ())
        motion_direction = str((track_state or {}).get('anchor_motion_direction', '') or '')
        if (
            'TRACK_LOST_IN_ZONE_A_TIMEOUT' in reasons
            and bool((track_state or {}).get('anchor_direction_locked'))
            and motion_direction in {'forward', 'reverse'}
        ):
            direction_code, direction_label = (
                (5, '正向前出') if motion_direction == 'forward' else (7, '反向前出')
            )
            track_state['direction_source'] = 'trajectory_inference'
            if not track_state.get('direction_fallback_traced'):
                track_state['direction_fallback_traced'] = True
                self.trace_record('direction_fallback', {
                    'trackId': int(track_state.get('track_id', 0) or 0),
                    'source': 'trajectory_inference',
                    'motionDirection': motion_direction,
                    'confidence': float(track_state.get('anchor_direction_confidence', 0.0) or 0.0),
                })
            return direction_code, direction_label
        if track_state is not None:
            track_state['direction_source'] = 'unknown'
        return 0, ''

    def water_contact(self, box, water_boxes):
        if not water_boxes or box is None:
            return False
        for wb in water_boxes:
            if box_iou(box, wb) >= 0.02:
                return True
        return False
