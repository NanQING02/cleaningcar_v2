from dataclasses import dataclass, field
from math import hypot
from time import time
from uuid import uuid4


@dataclass
class BusinessLifecycle:
    event_id: str
    tracker_ids: set = field(default_factory=set)
    stages: set = field(default_factory=set)
    vehicle_class: str = ''
    active_tracker_id: int = 0
    last_seen_ts: float = 0.0
    last_plate_ts: float = 0.0
    last_plate_box: tuple | None = None
    last_plate_text: str = ''
    last_plate_edge: str = ''
    last_motion_direction: str = 'unknown'
    lost_ts: float = 0.0
    closed: bool = False


class BusinessLifecycleManager:
    """Keeps business identity separate from short-lived tracker state."""

    def __init__(self, camera_id, grace_seconds=8.0):
        self.camera_id = str(camera_id)
        self.grace_seconds = max(0.1, float(grace_seconds))
        self.by_event_id = {}
        self.by_tracker_id = {}

    def create(self, tracker_id, vehicle_class, capture_ts=None):
        now = time() if capture_ts is None else float(capture_ts)
        event_id = f'{self.camera_id}-{int(now * 1000)}-{uuid4().hex[:10]}'
        lifecycle = BusinessLifecycle(
            event_id=event_id,
            tracker_ids={int(tracker_id)},
            vehicle_class=str(vehicle_class or ''),
            active_tracker_id=int(tracker_id),
            last_seen_ts=now,
        )
        self.by_event_id[event_id] = lifecycle
        self.by_tracker_id[int(tracker_id)] = event_id
        return lifecycle

    def get(self, tracker_id):
        event_id = self.by_tracker_id.get(int(tracker_id))
        return self.by_event_id.get(event_id) if event_id else None

    def touch(self, tracker_id, capture_ts=None, vehicle_class='', plate_text='', plate_box=None,
              plate_edge='', motion_direction='unknown'):
        lifecycle = self.get(tracker_id)
        if lifecycle is None:
            return None
        now = time() if capture_ts is None else float(capture_ts)
        lifecycle.last_seen_ts = now
        lifecycle.active_tracker_id = int(tracker_id)
        lifecycle.lost_ts = 0.0
        if vehicle_class and not lifecycle.vehicle_class:
            lifecycle.vehicle_class = str(vehicle_class)
        if plate_box is not None:
            lifecycle.last_plate_box = tuple(float(value) for value in plate_box)
            lifecycle.last_plate_ts = now
            lifecycle.last_plate_edge = str(plate_edge or '')
        if plate_text:
            lifecycle.last_plate_text = str(plate_text)
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
        lifecycle.active_tracker_id = 0
        lifecycle.lost_ts = time() if capture_ts is None else float(capture_ts)
        return lifecycle

    def finalize(self, tracker_id):
        lifecycle = self.get(tracker_id)
        if lifecycle is not None:
            lifecycle.closed = True
            lifecycle.active_tracker_id = 0
        return lifecycle

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

    def can_handoff(self, lifecycle, vehicle_class, capture_ts, plate_edge, plate_box,
                    has_valid_plate, motion_direction='unknown'):
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id:
            return False
        now = float(capture_ts)
        if now - lifecycle.lost_ts > self.grace_seconds or now - lifecycle.last_plate_ts > self.grace_seconds:
            return False
        if not has_valid_plate or not lifecycle.last_plate_edge:
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

    def find_handoff_candidates(self, vehicle_class, capture_ts, plate_edge, plate_box,
                                has_valid_plate, motion_direction='unknown'):
        return [
            lifecycle for lifecycle in self.by_event_id.values()
            if self.can_handoff(
                lifecycle, vehicle_class, capture_ts, plate_edge, plate_box,
                has_valid_plate, motion_direction,
            )
        ]

    def handoff(self, lifecycle, tracker_id, capture_ts):
        if lifecycle is None or lifecycle.closed or lifecycle.active_tracker_id:
            return None
        tracker_id = int(tracker_id)
        lifecycle.tracker_ids.add(tracker_id)
        lifecycle.active_tracker_id = tracker_id
        lifecycle.last_seen_ts = float(capture_ts)
        lifecycle.lost_ts = 0.0
        self.by_tracker_id[tracker_id] = lifecycle.event_id
        return lifecycle
