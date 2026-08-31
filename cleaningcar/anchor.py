from collections import deque
from dataclasses import asdict, dataclass
from math import hypot

from .vision import get_anchor_point


@dataclass(frozen=True)
class AnchorResult:
    track_id: int
    frame_idx: int
    mode: str
    selected_point: tuple
    legacy_point: tuple
    neutral_point: tuple
    directional_point: tuple
    adaptive_point: tuple
    raw_ground_point: tuple
    confidence: float
    box_truncated: bool
    truncated_edges: tuple
    reused_previous: bool
    vertical_offset_px: float
    legacy_flow_shift_px: float
    motion_direction: str
    direction_confidence: float
    direction_locked: bool
    direction_blend: float
    flow_projection_displacement_px: float

    def to_dict(self):
        data = asdict(self)
        data['selected_point'] = list(self.selected_point) if self.selected_point else None
        data['legacy_point'] = list(self.legacy_point) if self.legacy_point else None
        data['neutral_point'] = list(self.neutral_point) if self.neutral_point else None
        data['directional_point'] = list(self.directional_point) if self.directional_point else None
        data['adaptive_point'] = list(self.adaptive_point) if self.adaptive_point else None
        data['raw_ground_point'] = list(self.raw_ground_point) if self.raw_ground_point else None
        data['truncated_edges'] = list(self.truncated_edges)
        return data


class AnchorEstimator:
    def __init__(self, frame_size, flow_vector, logic_cfg=None):
        logic_cfg = logic_cfg or {}
        self.frame_w = max(1.0, float(frame_size[0]))
        self.frame_h = max(1.0, float(frame_size[1]))
        self.mode = str(logic_cfg.get('anchor_mode', 'directional') or 'directional').strip().lower()
        if self.mode not in {'legacy', 'directional'}:
            self.mode = 'directional'
        self.shadow_compare = bool(logic_cfg.get('anchor_shadow_compare', False))
        self.legacy_vertical_ratio = self._bounded_float(logic_cfg.get('anchor_offset_ratio', 0.1), 0.1, 0.0, 0.95)
        self.legacy_flow_shift_ratio = self._bounded_float(
            logic_cfg.get('anchor_legacy_flow_shift_ratio', 0.3),
            0.3,
            0.0,
            1.0,
        )
        self.adaptive_vertical_ratio = self._bounded_float(
            logic_cfg.get('anchor_adaptive_vertical_ratio', 0.08),
            0.08,
            0.0,
            0.5,
        )
        self.adaptive_vertical_cap_ratio = self._bounded_float(
            logic_cfg.get('anchor_adaptive_vertical_cap_ratio', 0.02),
            0.02,
            0.0,
            0.2,
        )
        self.edge_margin_ratio = self._bounded_float(
            logic_cfg.get('anchor_edge_margin_ratio', 0.01),
            0.01,
            0.0,
            0.1,
        )
        self.history_size = self._bounded_int(logic_cfg.get('anchor_history_size', 20), 20, 2, 200)
        self.reuse_max_frames = self._bounded_int(
            logic_cfg.get('anchor_reuse_max_frames', 12),
            12,
            0,
            300,
        )
        self.direction_window = self._bounded_int(
            logic_cfg.get('anchor_direction_window', 8),
            8,
            3,
            60,
        )
        self.direction_min_points = self._bounded_int(
            logic_cfg.get('anchor_direction_min_points', 5),
            5,
            3,
            self.direction_window,
        )
        self.direction_consistency = self._bounded_float(
            logic_cfg.get('anchor_direction_consistency', 0.7),
            0.7,
            0.5,
            1.0,
        )
        self.direction_min_displacement_ratio = self._bounded_float(
            logic_cfg.get('anchor_direction_min_displacement_ratio', 0.03),
            0.03,
            0.0,
            0.5,
        )
        self.direction_blend_frames = self._bounded_int(
            logic_cfg.get('anchor_direction_blend_frames', 5),
            5,
            1,
            30,
        )
        self._flow_unit = self._resolve_flow_unit(flow_vector)
        self._reliable_history = {}
        self._motion_history = {}
        self._direction_state = {}
        self._display_history = {}
        self._cache = {}

    @staticmethod
    def _bounded_float(value, default, minimum, maximum):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = float(default)
        return max(float(minimum), min(float(maximum), parsed))

    @staticmethod
    def _bounded_int(value, default, minimum, maximum):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = int(default)
        return max(int(minimum), min(int(maximum), parsed))

    @staticmethod
    def _resolve_flow_unit(flow_vector):
        try:
            start, end = flow_vector
            dx = float(end[0]) - float(start[0])
            dy = float(end[1]) - float(start[1])
        except Exception:
            return (0.0, 0.0)
        norm = hypot(dx, dy)
        if norm <= 1e-6:
            return (0.0, 0.0)
        return (dx / norm, dy / norm)

    def _clip_point(self, point):
        return (
            max(0.0, min(self.frame_w - 1.0, float(point[0]))),
            max(0.0, min(self.frame_h - 1.0, float(point[1]))),
        )

    def _truncated_edges(self, box):
        x1, y1, x2, y2 = [float(value) for value in box]
        margin_x = max(2.0, self.frame_w * self.edge_margin_ratio)
        margin_y = max(2.0, self.frame_h * self.edge_margin_ratio)
        edges = []
        if x1 <= margin_x:
            edges.append('left')
        if x2 >= self.frame_w - margin_x:
            edges.append('right')
        if y2 >= self.frame_h - margin_y:
            edges.append('bottom')
        if y1 <= margin_y:
            edges.append('top')
        return tuple(edges)

    def _predict_from_history(self, track_id, frame_idx):
        history = self._reliable_history.get(track_id)
        if not history:
            return None
        last_frame, last_point = history[-1]
        gap = max(0, int(frame_idx) - int(last_frame))
        if gap > self.reuse_max_frames:
            return None
        if len(history) < 2:
            return self._clip_point(last_point)
        prev_frame, prev_point = history[-2]
        frame_delta = max(1, int(last_frame) - int(prev_frame))
        vx = (float(last_point[0]) - float(prev_point[0])) / frame_delta
        vy = (float(last_point[1]) - float(prev_point[1])) / frame_delta
        max_step = max(4.0, self.frame_h * 0.04) * max(1, gap)
        dx = vx * gap
        dy = vy * gap
        step = hypot(dx, dy)
        if step > max_step and step > 1e-6:
            scale = max_step / step
            dx *= scale
            dy *= scale
        return self._clip_point((last_point[0] + dx, last_point[1] + dy))

    def _update_direction(self, track_id, frame_idx, neutral_point, box_height):
        if track_id <= 0 or self._flow_unit == (0.0, 0.0):
            return 'unknown', 0.0, False, 0.0, 0.0

        history = self._motion_history.setdefault(
            track_id,
            deque(maxlen=self.direction_window),
        )
        history.append((int(frame_idx), neutral_point))
        state = self._direction_state.setdefault(
            track_id,
            {'direction': 'unknown', 'confidence': 0.0, 'blend_steps': 0},
        )
        flow_x, flow_y = self._flow_unit
        projections = [float(point[0]) * flow_x + float(point[1]) * flow_y for _, point in history]
        displacement = projections[-1] - projections[0] if len(projections) >= 2 else 0.0

        if state['direction'] == 'unknown' and len(projections) >= self.direction_min_points:
            minimum_displacement = max(8.0, self.direction_min_displacement_ratio * box_height)
            meaningful_steps = [
                current - previous
                for previous, current in zip(projections, projections[1:])
                if abs(current - previous) >= 1.0
            ]
            if abs(displacement) >= minimum_displacement and meaningful_steps:
                expected_positive = displacement > 0.0
                consistent_steps = sum((step > 0.0) == expected_positive for step in meaningful_steps)
                consistency = consistent_steps / len(meaningful_steps)
                if consistency >= self.direction_consistency:
                    state['direction'] = 'forward' if expected_positive else 'reverse'
                    state['confidence'] = float(consistency)
                    state['blend_steps'] = 0

        direction = str(state['direction'])
        if direction == 'unknown':
            return direction, 0.0, False, 0.0, float(displacement)

        if state['blend_steps'] < self.direction_blend_frames:
            state['blend_steps'] += 1
        blend = min(1.0, float(state['blend_steps']) / self.direction_blend_frames)
        return direction, float(state['confidence']), True, blend, float(displacement)

    def estimate(self, track_id, box, frame_idx):
        if not box or len(box) != 4:
            return None
        try:
            normalized_track_id = int(track_id or 0)
            normalized_frame_idx = int(frame_idx)
            parsed_box = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            return None
        cache_key = (normalized_track_id, normalized_frame_idx)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        x1, y1, x2, y2 = parsed_box
        box_height = max(1.0, y2 - y1)
        neutral_point = self._clip_point(get_anchor_point(parsed_box, self.legacy_vertical_ratio))
        flow_x, flow_y = self._flow_unit
        legacy_shift = self.legacy_flow_shift_ratio * box_height
        legacy_point = self._clip_point((
            neutral_point[0] - flow_x * legacy_shift,
            neutral_point[1] - flow_y * legacy_shift,
        ))
        motion_direction, direction_confidence, direction_locked, direction_blend, displacement = (
            self._update_direction(
                normalized_track_id,
                normalized_frame_idx,
                neutral_point,
                box_height,
            )
        )
        direction_sign = 1.0 if motion_direction == 'forward' else -1.0
        if not direction_locked:
            directional_point = neutral_point
        else:
            directional_point = self._clip_point((
                neutral_point[0] - flow_x * legacy_shift * direction_sign * direction_blend,
                neutral_point[1] - flow_y * legacy_shift * direction_sign * direction_blend,
            ))

        vertical_cap = self.frame_h * self.adaptive_vertical_cap_ratio
        vertical_offset = min(self.adaptive_vertical_ratio * box_height, vertical_cap)
        raw_ground_point = self._clip_point((0.5 * (x1 + x2), y2 - vertical_offset))
        truncated_edges = self._truncated_edges(parsed_box)
        box_truncated = bool(truncated_edges)
        reused_previous = False
        adaptive_point = raw_ground_point
        confidence = 1.0
        if box_truncated:
            predicted = self._predict_from_history(normalized_track_id, normalized_frame_idx)
            if predicted is not None:
                adaptive_point = predicted
                reused_previous = True
                last_frame = self._reliable_history[normalized_track_id][-1][0]
                gap = max(0, normalized_frame_idx - int(last_frame))
                confidence = max(0.4, 0.75 - 0.04 * gap)
            else:
                confidence = 0.35
        elif normalized_track_id > 0:
            history = self._reliable_history.setdefault(
                normalized_track_id,
                deque(maxlen=self.history_size),
            )
            history.append((normalized_frame_idx, adaptive_point))

        if self.mode == 'directional':
            selected_point = directional_point
        else:
            selected_point = legacy_point
        result = AnchorResult(
            track_id=normalized_track_id,
            frame_idx=normalized_frame_idx,
            mode=self.mode,
            selected_point=selected_point,
            legacy_point=legacy_point,
            neutral_point=neutral_point,
            directional_point=directional_point,
            adaptive_point=adaptive_point,
            raw_ground_point=raw_ground_point,
            confidence=float(confidence),
            box_truncated=box_truncated,
            truncated_edges=truncated_edges,
            reused_previous=reused_previous,
            vertical_offset_px=float(vertical_offset),
            legacy_flow_shift_px=float(legacy_shift),
            motion_direction=motion_direction,
            direction_confidence=direction_confidence,
            direction_locked=direction_locked,
            direction_blend=direction_blend,
            flow_projection_displacement_px=displacement,
        )
        self._cache[cache_key] = result
        if normalized_track_id > 0:
            display = self._display_history.setdefault(
                normalized_track_id,
                deque(maxlen=self.history_size),
            )
            display.append((
                normalized_frame_idx,
                legacy_point,
                neutral_point,
                directional_point,
                adaptive_point,
            ))
        return result

    def get_history(self, track_id):
        try:
            normalized_track_id = int(track_id)
        except (TypeError, ValueError):
            return []
        return list(self._display_history.get(normalized_track_id, ()))

    def cleanup(self, frame_idx, max_age):
        cutoff = int(frame_idx) - max(1, int(max_age))
        for store in (self._reliable_history, self._display_history):
            for track_id in list(store.keys()):
                history = store.get(track_id)
                if not history or int(history[-1][0]) < cutoff:
                    store.pop(track_id, None)
        for track_id in list(self._motion_history.keys()):
            history = self._motion_history.get(track_id)
            if not history or int(history[-1][0]) < cutoff:
                self._motion_history.pop(track_id, None)
                self._direction_state.pop(track_id, None)
        for key in list(self._cache.keys()):
            if int(key[1]) < cutoff:
                self._cache.pop(key, None)
