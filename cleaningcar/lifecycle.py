from dataclasses import dataclass, field
from math import hypot
from time import time
from uuid import uuid4

from utils.upload_queue import PLATFORM_EVENT_ID_MAX_LENGTH, normalize_event_id


@dataclass
class BusinessLifecycle:
    event_id: str
    tracker_ids: set = field(default_factory=set)
    stages: set = field(default_factory=set)
    vehicle_class: str = ''
    active_tracker_id: int = 0
    last_tracker_id: int = 0
    last_seen_ts: float = 0.0
    last_plate_ts: float = 0.0
    last_plate_box: tuple | None = None
    last_plate_text: str = ''
    last_plate_color: str = ''
    last_plate_color_conf: float = 0.0
    last_plate_type: str = ''
    last_plate_edge: str = ''
    last_motion_direction: str = 'unknown'
    lost_ts: float = 0.0
    closed: bool = False
    closed_ts: float = 0.0


class BusinessLifecycleManager:
    """Keeps business identity separate from short-lived tracker state."""

    EVENT_ID_MAX_LENGTH = PLATFORM_EVENT_ID_MAX_LENGTH
    EVENT_ID_CAMERA_PREFIX_LENGTH = 11

    def __init__(self, camera_id, grace_seconds=4.0):
        self.camera_id = str(camera_id)
        self.grace_seconds = max(0.1, float(grace_seconds))
        self.by_event_id = {}
        self.by_tracker_id = {}

    def _new_event_id(self, capture_ts):
        camera_key = ''.join(
            char for char in self.camera_id
            if char.isalnum() or char in {'-', '_'}
        )[:self.EVENT_ID_CAMERA_PREFIX_LENGTH].strip('-_') or 'CAM'
        event_id = f'{camera_key}-{int(float(capture_ts) * 1000)}-{uuid4().hex[:10]}'
        return normalize_event_id(event_id, self.EVENT_ID_MAX_LENGTH)

    def create(self, tracker_id, vehicle_class, capture_ts=None):
        now = time() if capture_ts is None else float(capture_ts)
        event_id = self._new_event_id(now)
        lifecycle = BusinessLifecycle(
            event_id=event_id,
            tracker_ids={int(tracker_id)},
            vehicle_class=str(vehicle_class or ''),
            active_tracker_id=int(tracker_id),
            last_tracker_id=int(tracker_id),
            last_seen_ts=now,
        )
        self.by_event_id[event_id] = lifecycle
        self.by_tracker_id[int(tracker_id)] = event_id
        return lifecycle

    def get(self, tracker_id):
        event_id = self.by_tracker_id.get(int(tracker_id))
        return self.by_event_id.get(event_id) if event_id else None

    def touch(self, tracker_id, capture_ts=None, vehicle_class='', plate_text='', plate_box=None,
              plate_edge='', motion_direction='unknown', plate_color='',
              plate_color_conf=0.0, plate_type=''):
        lifecycle = self.get(tracker_id)
        if lifecycle is None:
            return None
        now = time() if capture_ts is None else float(capture_ts)
        lifecycle.last_seen_ts = now
        lifecycle.active_tracker_id = int(tracker_id)
        lifecycle.last_tracker_id = int(tracker_id)
        lifecycle.lost_ts = 0.0
        if vehicle_class and not lifecycle.vehicle_class:
            lifecycle.vehicle_class = str(vehicle_class)
        if plate_box is not None:
            lifecycle.last_plate_box = tuple(float(value) for value in plate_box)
            lifecycle.last_plate_ts = now
            lifecycle.last_plate_edge = str(plate_edge or '')
        if plate_text:
            plate_text = str(plate_text)
            if lifecycle.last_plate_text and lifecycle.last_plate_text != plate_text:
                lifecycle.last_plate_color = ''
                lifecycle.last_plate_color_conf = 0.0
                lifecycle.last_plate_type = ''
            lifecycle.last_plate_text = plate_text
            if plate_color:
                lifecycle.last_plate_color = str(plate_color)
                try:
                    lifecycle.last_plate_color_conf = float(plate_color_conf or 0.0)
                except (TypeError, ValueError):
                    lifecycle.last_plate_color_conf = 0.0
            if plate_type:
                lifecycle.last_plate_type = str(plate_type)
        if motion_direction in {'forward', 'reverse'}:
            lifecycle.last_motion_direction = motion_direction
        return lifecycle

    def mark_stage(self, tracker_id, stage):
        lifecycle = self.get(tracker_id)
        if lifecycle is not None:
            lifecycle.stages.add(int(stage))
        return lifecycle

    def mark_lost(self, tracker_id, capture_ts=None):
        lifecycle = self.get(tracker_id)
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id != int(tracker_id):
            return lifecycle
        lifecycle.last_tracker_id = int(tracker_id)
        lifecycle.active_tracker_id = 0
        lifecycle.lost_ts = time() if capture_ts is None else float(capture_ts)
        return lifecycle

    def finalize(self, tracker_id, capture_ts=None):
        lifecycle = self.get(tracker_id)
        if lifecycle is not None:
            if not lifecycle.closed or lifecycle.closed_ts <= 0.0:
                lifecycle.closed_ts = time() if capture_ts is None else float(capture_ts)
            lifecycle.closed = True
            lifecycle.active_tracker_id = 0
        return lifecycle

    def cleanup(self, capture_ts=None, closed_retention_seconds=60.0):
        now = time() if capture_ts is None else float(capture_ts)
        retention = max(self.grace_seconds, float(closed_retention_seconds))
        expired_event_ids = [
            event_id
            for event_id, lifecycle in self.by_event_id.items()
            if (
                lifecycle.closed
                and lifecycle.closed_ts > 0.0
                and now - lifecycle.closed_ts >= retention
            )
        ]
        for event_id in expired_event_ids:
            lifecycle = self.by_event_id.pop(event_id, None)
            if lifecycle is None:
                continue
            for tracker_id in lifecycle.tracker_ids:
                if self.by_tracker_id.get(int(tracker_id)) == event_id:
                    self.by_tracker_id.pop(int(tracker_id), None)
        return len(expired_event_ids)

    def snapshot(self):
        lifecycles = list(self.by_event_id.values())
        return {
            'total': len(lifecycles),
            'active': sum(1 for item in lifecycles if not item.closed and item.active_tracker_id),
            'waiting': sum(1 for item in lifecycles if not item.closed and not item.active_tracker_id),
            'closed_retained': sum(1 for item in lifecycles if item.closed),
            'tracker_mappings': len(self.by_tracker_id),
        }

    def is_waiting(self, tracker_id, capture_ts):
        lifecycle = self.get(tracker_id)
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id:
            return False
        return 0.0 <= float(capture_ts) - lifecycle.lost_ts <= self.grace_seconds

    @staticmethod
    def _boxes_continuous(previous_box, current_box):
        if previous_box is None or current_box is None:
            return False
        px = 0.5 * (previous_box[0] + previous_box[2])
        py = 0.5 * (previous_box[1] + previous_box[3])
        cx = 0.5 * (current_box[0] + current_box[2])
        cy = 0.5 * (current_box[1] + current_box[3])
        previous_size = max(previous_box[2] - previous_box[0], previous_box[3] - previous_box[1], 1.0)
        current_size = max(current_box[2] - current_box[0], current_box[3] - current_box[1], 1.0)
        return hypot(cx - px, cy - py) <= max(96.0, 4.0 * max(previous_size, current_size))

    def can_handoff(self, lifecycle, vehicle_class, capture_ts, plate_text, plate_edge, plate_box,
                    has_valid_plate, motion_direction='unknown'):
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id:
            return False
        now = float(capture_ts)
        if now - lifecycle.lost_ts > self.grace_seconds or now - lifecycle.last_plate_ts > self.grace_seconds:
            return False
        if not has_valid_plate or not lifecycle.last_plate_edge:
            return False
        if not plate_text or str(plate_text) != lifecycle.last_plate_text:
            return False
        if lifecycle.vehicle_class != str(vehicle_class or ''):
            return False
        if lifecycle.last_plate_edge != str(plate_edge or ''):
            return False
        if not self._boxes_continuous(lifecycle.last_plate_box, plate_box):
            return False
        return not (
            motion_direction in {'forward', 'reverse'}
            and lifecycle.last_motion_direction in {'forward', 'reverse'}
            and motion_direction != lifecycle.last_motion_direction
        )

    def find_handoff_candidates(self, vehicle_class, capture_ts, plate_text, plate_edge, plate_box,
                                has_valid_plate, motion_direction='unknown'):
        return [
            lifecycle for lifecycle in self.by_event_id.values()
            if self.can_handoff(
                lifecycle, vehicle_class, capture_ts, plate_text, plate_edge, plate_box,
                has_valid_plate, motion_direction,
            )
        ]

    def find_waiting_lifecycles(self, vehicle_class, capture_ts):
        vehicle_class = str(vehicle_class or '')
        now = float(capture_ts)
        return [
            lifecycle
            for lifecycle in self.by_event_id.values()
            if (
                not lifecycle.closed
                and not lifecycle.active_tracker_id
                and 0.0 <= now - lifecycle.lost_ts <= self.grace_seconds
                and lifecycle.vehicle_class == vehicle_class
                and bool(lifecycle.last_plate_text)
                and bool(lifecycle.last_plate_edge)
                and lifecycle.last_plate_box is not None
            )
        ]

    def handoff(self, lifecycle, tracker_id, capture_ts):
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id:
            return None
        tracker_id = int(tracker_id)
        lifecycle.tracker_ids.add(tracker_id)
        lifecycle.active_tracker_id = tracker_id
        lifecycle.last_tracker_id = tracker_id
        lifecycle.last_seen_ts = float(capture_ts)
        lifecycle.lost_ts = 0.0
        self.by_tracker_id[tracker_id] = lifecycle.event_id
        return lifecycle
