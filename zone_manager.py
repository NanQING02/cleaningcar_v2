from dataclasses import dataclass, field
from math import isfinite
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def point_in_polygon(point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    contour = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


@dataclass
class ZoneState:
    inside_a: bool = False
    inside_b: bool = False
    zone_a_state: str = 'UNSEEN'
    zone_a_region: str = 'INVALID'
    signed_distance: Optional[float] = None
    dynamic_margin: float = 0.0
    observed_outside_count: int = 0
    enter_core_count: int = 0
    exit_outside_count: int = 0
    initial_core_compat: bool = False
    born_inside_a: bool = False
    born_inside_pending: bool = False
    born_inside_outcome: str = ''
    transition_reason: str = 'initialized'
    last_observation_frame: int = -1
    enter_ratio: Optional[float] = None
    exit_ratio: Optional[float] = None
    entry_point: Tuple[float, float] = (0.0, 0.0)
    exit_point: Tuple[float, float] = (0.0, 0.0)
    last_anchor: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ZoneManager:
    zone_a: List[Tuple[float, float]]
    zone_b: List[Tuple[float, float]]
    flow_vector: Tuple[Tuple[float, float], Tuple[float, float]]
    direction_reference_edge: object = 'auto'
    direction_split_ratio: float = 0.5
    entry_hysteresis: int = 3
    exit_hysteresis: int = 3
    zone_a_margin_ratio: float = 0.10
    zone_a_margin_min_px: float = 4.0
    zone_a_margin_max_px: float = 24.0
    zone_a_observed_outside_hits: int = 3
    zone_a_enter_core_hits: int = 3
    zone_a_exit_outside_hits: int = 5
    state_cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.entry_hysteresis = max(1, int(self.entry_hysteresis))
        self.exit_hysteresis = max(1, int(self.exit_hysteresis))
        self.zone_a_margin_ratio = max(0.0, float(self.zone_a_margin_ratio))
        self.zone_a_margin_min_px = max(0.0, float(self.zone_a_margin_min_px))
        self.zone_a_margin_max_px = max(self.zone_a_margin_min_px, float(self.zone_a_margin_max_px))
        self.zone_a_observed_outside_hits = max(1, int(self.zone_a_observed_outside_hits))
        self.zone_a_enter_core_hits = max(1, int(self.zone_a_enter_core_hits))
        self.zone_a_exit_outside_hits = max(1, int(self.zone_a_exit_outside_hits))
        self.direction_split_ratio = max(0.0, min(1.0, float(self.direction_split_ratio)))
        self._zone_a_contour = np.array(self.zone_a, dtype=np.float32) if len(self.zone_a) >= 3 else None
        self._direction_edge_index = self._resolve_direction_reference_edge()
        self._direction_axis = None
        self._direction_min = 0.0
        self._direction_max = 0.0
        self._configure_direction_axis()

    @property
    def direction_reference_edge_index(self) -> Optional[int]:
        return self._direction_edge_index

    def _flow_unit(self) -> Optional[Tuple[float, float]]:
        flow_start, flow_end = self.flow_vector
        fx = float(flow_end[0]) - float(flow_start[0])
        fy = float(flow_end[1]) - float(flow_start[1])
        norm = float(np.hypot(fx, fy))
        if norm <= 1e-6:
            return None
        return fx / norm, fy / norm

    def _resolve_direction_reference_edge(self) -> Optional[int]:
        edge_count = len(self.zone_a)
        if edge_count < 2:
            return None
        raw = self.direction_reference_edge
        if not isinstance(raw, str) or raw.strip().lower() != 'auto':
            try:
                index = int(raw)
            except (TypeError, ValueError):
                index = -1
            if 0 <= index < edge_count:
                return index
        flow_unit = self._flow_unit()
        if flow_unit is None:
            return 0
        fx, fy = flow_unit
        candidates = []
        for index, start in enumerate(self.zone_a):
            end = self.zone_a[(index + 1) % edge_count]
            ex = float(end[0]) - float(start[0])
            ey = float(end[1]) - float(start[1])
            length = float(np.hypot(ex, ey))
            if length <= 1e-6:
                continue
            perpendicular_error = abs((ex / length) * fx + (ey / length) * fy)
            candidates.append((perpendicular_error, -length, index))
        return min(candidates)[2] if candidates else 0

    def _configure_direction_axis(self):
        index = self._direction_edge_index
        flow_unit = self._flow_unit()
        if index is None or flow_unit is None or len(self.zone_a) < 2:
            return
        start = self.zone_a[index]
        end = self.zone_a[(index + 1) % len(self.zone_a)]
        ex = float(end[0]) - float(start[0])
        ey = float(end[1]) - float(start[1])
        length = float(np.hypot(ex, ey))
        if length <= 1e-6:
            return
        nx, ny = -ey / length, ex / length
        if nx * flow_unit[0] + ny * flow_unit[1] < 0.0:
            nx, ny = -nx, -ny
        projections = [float(pt[0]) * nx + float(pt[1]) * ny for pt in self.zone_a]
        if not projections:
            return
        lo = min(projections)
        hi = max(projections)
        if hi - lo <= 1e-6:
            return
        self._direction_axis = (nx, ny)
        self._direction_min = lo
        self._direction_max = hi

    def update_track(
        self,
        track_id: int,
        anchor_point: Tuple[float, float],
        frame_idx: int,
        vehicle_height: Optional[float] = None,
    ):
        st = self.state_cache.setdefault(track_id, {
            'state': ZoneState(),
            'b_entry_counter': 0,
            'b_exit_counter': 0,
            'last_anchor': anchor_point,
            'last_frame': frame_idx,
            'last_zone_a_frame': -1,
            'last_zone_b_frame': -1,
            'last_clear_region': '',
            'first_valid_region': '',
            'outside_confirmed_before_entry': False,
        })
        st['last_frame'] = frame_idx
        state: ZoneState = st['state']
        flags = {'enter_a': False, 'exit_a': False, 'enter_b': False, 'exit_b': False}
        if not self._valid_anchor(anchor_point):
            state.zone_a_region = 'INVALID'
            state.signed_distance = None
            state.transition_reason = 'invalid_anchor'
            return state, flags

        normalized_anchor = (float(anchor_point[0]), float(anchor_point[1]))
        st['last_anchor'] = normalized_anchor
        state.last_anchor = normalized_anchor

        inside_b = point_in_polygon(normalized_anchor, self.zone_b)
        if st['last_zone_b_frame'] != frame_idx:
            st['last_zone_b_frame'] = frame_idx
            st['b_entry_counter'] = min(
                self.entry_hysteresis,
                st['b_entry_counter'] + 1,
            ) if inside_b else 0
            st['b_exit_counter'] = min(
                self.exit_hysteresis,
                st['b_exit_counter'] + 1,
            ) if not inside_b else 0

            if st['b_entry_counter'] >= self.entry_hysteresis and not state.inside_b:
                state.inside_b = True
                flags['enter_b'] = True
            elif st['b_exit_counter'] >= self.exit_hysteresis and state.inside_b:
                state.inside_b = False
                flags['exit_b'] = True

        if self._zone_a_contour is None or not self._valid_vehicle_height(vehicle_height):
            state.zone_a_region = 'INVALID'
            state.signed_distance = None
            state.dynamic_margin = 0.0
            state.transition_reason = (
                'invalid_zone_a_polygon' if self._zone_a_contour is None else 'invalid_vehicle_height'
            )
            return state, flags
        if st['last_zone_a_frame'] == frame_idx:
            state.transition_reason = 'duplicate_frame_ignored'
            return state, flags
        st['last_zone_a_frame'] = frame_idx
        state.last_observation_frame = int(frame_idx)

        margin = min(
            self.zone_a_margin_max_px,
            max(self.zone_a_margin_min_px, float(vehicle_height) * self.zone_a_margin_ratio),
        )
        signed_distance = float(cv2.pointPolygonTest(
            self._zone_a_contour,
            normalized_anchor,
            True,
        ))
        if signed_distance > margin:
            region = 'CORE'
        elif signed_distance < -margin:
            region = 'OUTSIDE'
        else:
            region = 'BUFFER'

        state.zone_a_region = region
        state.signed_distance = signed_distance
        state.dynamic_margin = margin
        if not st['first_valid_region']:
            st['first_valid_region'] = region

        if region == 'CORE':
            st['last_clear_region'] = 'CORE'
            state.observed_outside_count = 0
            state.exit_outside_count = 0
            if state.inside_a:
                state.enter_core_count = self.zone_a_enter_core_hits
                state.zone_a_state = 'INSIDE_A_CANDIDATE'
                state.transition_reason = 'stable_inside_core'
            else:
                state.enter_core_count = min(
                    self.zone_a_enter_core_hits,
                    state.enter_core_count + 1,
                )
                state.transition_reason = (
                    f'core_observation_{state.enter_core_count}_of_{self.zone_a_enter_core_hits}'
                )
                if state.enter_core_count >= self.zone_a_enter_core_hits:
                    state.inside_a = True
                    state.zone_a_state = 'INSIDE_A_CANDIDATE'
                    state.initial_core_compat = bool(
                        st['first_valid_region'] == 'CORE'
                        and not st['outside_confirmed_before_entry']
                    )
                    state.born_inside_a = state.initial_core_compat
                    state.born_inside_pending = state.initial_core_compat
                    state.born_inside_outcome = (
                        'CANDIDATE' if state.initial_core_compat else ''
                    )
                    state.enter_ratio = self._relative_position(normalized_anchor)
                    state.entry_point = normalized_anchor
                    state.exit_ratio = None
                    state.exit_point = (0.0, 0.0)
                    state.transition_reason = (
                        'born_inside_a_candidate'
                        if state.initial_core_compat
                        else 'enter_a_core_confirmed'
                    )
                    flags['enter_a'] = True
        elif region == 'OUTSIDE':
            st['last_clear_region'] = 'OUTSIDE'
            state.enter_core_count = 0
            if state.inside_a:
                state.observed_outside_count = 0
                state.exit_outside_count = min(
                    self.zone_a_exit_outside_hits,
                    state.exit_outside_count + 1,
                )
                state.transition_reason = (
                    f'exit_observation_{state.exit_outside_count}_of_{self.zone_a_exit_outside_hits}'
                )
                if state.exit_outside_count >= self.zone_a_exit_outside_hits:
                    state.inside_a = False
                    state.zone_a_state = 'OBSERVED_OUTSIDE'
                    state.observed_outside_count = self.zone_a_observed_outside_hits
                    state.exit_ratio = self._relative_position(normalized_anchor)
                    state.exit_point = normalized_anchor
                    state.transition_reason = 'exit_a_outside_confirmed'
                    flags['exit_a'] = True
            else:
                state.exit_outside_count = 0
                state.observed_outside_count = min(
                    self.zone_a_observed_outside_hits,
                    state.observed_outside_count + 1,
                )
                state.transition_reason = (
                    f'outside_observation_{state.observed_outside_count}_of_'
                    f'{self.zone_a_observed_outside_hits}'
                )
                if state.observed_outside_count >= self.zone_a_observed_outside_hits:
                    state.zone_a_state = 'OBSERVED_OUTSIDE'
                    st['outside_confirmed_before_entry'] = True
                    state.transition_reason = 'observed_outside_confirmed'
        else:
            if st['last_clear_region'] == 'CORE':
                state.observed_outside_count = 0
                state.exit_outside_count = 0
                state.transition_reason = 'buffer_after_core'
            elif st['last_clear_region'] == 'OUTSIDE':
                state.enter_core_count = 0
                state.transition_reason = 'buffer_after_outside'
            else:
                state.transition_reason = 'buffer_without_clear_region'

        if flags['enter_b'] and state.born_inside_pending:
            state.born_inside_pending = False
            state.born_inside_outcome = 'PROMOTED'
            state.transition_reason = 'born_inside_promoted_by_zone_b'
        elif flags['exit_a'] and state.born_inside_pending:
            state.born_inside_pending = False
            state.born_inside_outcome = 'PASS_BY'
            state.transition_reason = 'born_inside_pass_by'

        return state, flags

    @staticmethod
    def _valid_anchor(anchor_point) -> bool:
        if not isinstance(anchor_point, (list, tuple)) or len(anchor_point) < 2:
            return False
        try:
            return isfinite(float(anchor_point[0])) and isfinite(float(anchor_point[1]))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _valid_vehicle_height(vehicle_height) -> bool:
        try:
            value = float(vehicle_height)
        except (TypeError, ValueError):
            return False
        return isfinite(value) and value > 0.0

    def cleanup(self, current_frame: int, max_age: int):
        for track_id in list(self.state_cache.keys()):
            if current_frame - self.state_cache[track_id]['last_frame'] > max_age:
                self.state_cache.pop(track_id, None)

    def drop_track(self, track_id: int):
        self.state_cache.pop(track_id, None)

    def resolve_direction(self, state: ZoneState) -> Tuple[int, str]:
        if state is None:
            return 0, ''
        flow_start, flow_end = self.flow_vector
        fx = flow_end[0] - flow_start[0]
        fy = flow_end[1] - flow_start[1]
        if fx == 0 and fy == 0:
            return 0, ''
        entry_ratio = state.enter_ratio
        exit_ratio = state.exit_ratio
        if entry_ratio is None and state.entry_point != (0.0, 0.0):
            entry_ratio = self._relative_position(state.entry_point)
        exit_pt = state.exit_point if state.exit_point != (0.0, 0.0) else state.last_anchor
        if exit_ratio is None and exit_pt != (0.0, 0.0):
            exit_ratio = self._relative_position(exit_pt)
        if entry_ratio is None or exit_ratio is None:
            return 0, ''
        mid = self.direction_split_ratio
        entry_forward = entry_ratio <= mid
        exit_front = exit_ratio >= mid
        if entry_forward and exit_front:
            return 5, '正向前出'
        if (not entry_forward) and exit_front:
            return 7, '反向前出'
        if entry_forward and (not exit_front):
            return 6, '正向反出'
        return 8, '反向反出'

    def _relative_position(self, point: Tuple[float, float]) -> float:
        if self._direction_axis is not None:
            nx, ny = self._direction_axis
            span = self._direction_max - self._direction_min
            if span > 1e-6:
                projection = float(point[0]) * nx + float(point[1]) * ny
                ratio = (projection - self._direction_min) / span
                return max(0.0, min(1.0, ratio))
        flow_start, flow_end = self.flow_vector
        fx = flow_end[0] - flow_start[0]
        fy = flow_end[1] - flow_start[1]
        seg_len_sq = fx * fx + fy * fy
        if seg_len_sq <= 1e-6:
            return 0.0
        px = point[0] - flow_start[0]
        py = point[1] - flow_start[1]
        proj = (px * fx + py * fy) / seg_len_sq
        return max(0.0, min(1.0, proj))
