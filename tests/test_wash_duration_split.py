import tempfile
import unittest

from cleaningcar.events import EventManager


class _ZoneState:
    def __init__(self, inside_a=False, inside_b=False):
        self.inside_a = inside_a
        self.inside_b = inside_b
        self.born_inside_a = False
        self.born_inside_pending = False
        self.born_inside_outcome = ''


class _ScriptedZoneManager:
    """可脚本控制Zone状态的zone manager替身。"""

    def __init__(self):
        self.state = _ZoneState()
        self.flags = {'enter_a': False, 'exit_a': False, 'enter_b': False, 'exit_b': False}

    def set(self, inside_b=None, enter_b=False):
        if inside_b is not None:
            self.state.inside_b = inside_b
        self.flags = {
            'enter_a': False,
            'exit_a': False,
            'enter_b': bool(enter_b),
            'exit_b': False,
        }

    def update_track(self, track_id, anchor_point, frame_idx, vehicle_height=None):
        return self.state, dict(self.flags)

    def drop_track(self, track_id):
        return None

    def resolve_direction(self, state):
        return 0, ''


class _CollectingUploader:
    def __init__(self):
        self.enqueued = []

    def enqueue(self, payload):
        self.enqueued.append(dict(payload))


class WashDurationSplitTests(unittest.TestCase):
    def _manager(self, uploader=None):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.zone = _ScriptedZoneManager()
        config = {
            "logic": {},
            "shadow_pool": {},
            "event_capture_dir": self._tmp.name,
            "event_output_dir": self._tmp.name,
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=self.zone,
            uploader=uploader,
        )

    def _update(self, mgr, frame_idx, water_boxes, manual=None, table=None, cleaning_label=''):
        mgr.update_track(
            track_id=1,
            plate_box=None,
            vehicle_box=[10, 10, 50, 60],
            plate_text='',
            frame_idx=frame_idx,
            frame=None,
            water_boxes=water_boxes,
            water_active=bool(water_boxes),
            is_plate=False,
            vehicle_label='car',
            vehicle_conf=0.9,
            plate_conf=None,
            confirmed=False,
            cleaning_label=cleaning_label,
            manual_detected=manual,
            table_detected=table,
        )

    def _ensure_track(self, mgr, frame_idx=0):
        """先空跑一次update_track，建立含全部默认键的track_state。"""
        self._update(mgr, frame_idx, water_boxes=[])
        return mgr.tracks[1]

    def _drive_counting(self, mgr, manual, table, frames=5, start=1):
        # 直接置为type2已达标，专测计数与payload，不驱动完整type1-4链路
        st = self._ensure_track(mgr)
        st['type2_qualified'] = True
        self.zone.set(inside_b=True)
        for frame_idx in range(start, start + frames):
            self._update(
                mgr,
                frame_idx,
                water_boxes=[[1, 1, 20, 20]],
                manual=manual,
                table=table,
                cleaning_label='manual' if manual else 'cleaning table',
            )
        return st

    def test_split_counters_count_independently(self):
        mgr = self._manager()
        st = self._drive_counting(mgr, manual=True, table=False, frames=5)
        self.assertEqual(st['manual_wash_frames'], 5)
        self.assertEqual(st['table_wash_frames'], 0)
        self.assertEqual(st['effective_wash_frames'], 5)

    def test_same_frame_both_classes_counts_each_once(self):
        mgr = self._manager()
        st = self._drive_counting(mgr, manual=True, table=True, frames=3)
        # 同帧两类并存各计一次；总时长按并集只计一次
        self.assertEqual(st['manual_wash_frames'], 3)
        self.assertEqual(st['table_wash_frames'], 3)
        self.assertEqual(st['effective_wash_frames'], 3)

    def test_cleaning_label_fallback_derives_class(self):
        mgr = self._manager()
        st = self._ensure_track(mgr)
        st['type2_qualified'] = True
        self.zone.set(inside_b=True)
        self._update(mgr, 1, water_boxes=[[1, 1, 20, 20]], cleaning_label='manual')
        self._update(mgr, 2, water_boxes=[[1, 1, 20, 20]], cleaning_label='cleaning table')

        self.assertEqual(st['manual_wash_frames'], 1)
        self.assertEqual(st['table_wash_frames'], 1)

    def test_type5_payload_and_event_carry_split_durations(self):
        uploader = _CollectingUploader()
        mgr = self._manager(uploader=uploader)
        st = self._drive_counting(mgr, manual=True, table=True, frames=5)
        st['wash_duration'] = 0.2

        event = {
            'id': 'evt-1',
            'type': 5,
            'captureTime': '2026-09-29 10:00:00',
            'trackId': 1,
        }
        payload = mgr._build_api_payload(event, st, 10)

        self.assertEqual(payload['manualWashDuration'], 0.2)
        self.assertEqual(payload['cleaningTableWashDuration'], 0.2)
        self.assertEqual(payload['totalWashDuration'], 0.2)
        # 分时长只在type5出现
        event_t2 = dict(event, type=2)
        payload_t2 = mgr._build_api_payload(event_t2, st, 10)
        self.assertNotIn('manualWashDuration', payload_t2)
        self.assertNotIn('cleaningTableWashDuration', payload_t2)

    def test_reentering_zone_b_resets_split_counters(self):
        mgr = self._manager()
        st = self._drive_counting(mgr, manual=True, table=False, frames=3)

        # 车辆再次进入Zone B：分计数随effective_wash_frames一起清零
        self.zone.set(inside_b=True, enter_b=True)
        self._update(mgr, 10, water_boxes=[[1, 1, 20, 20]], manual=True, cleaning_label='manual')

        self.assertEqual(st['manual_wash_frames'], 1)
        self.assertEqual(st['effective_wash_frames'], 1)
        self.assertEqual(st['water_window_hits'], 1)

    def test_water_outside_zone_b_not_counted(self):
        mgr = self._manager()
        st = self._ensure_track(mgr)
        st['type2_qualified'] = True
        self.zone.set(inside_b=False)
        self._update(mgr, 1, water_boxes=[[1, 1, 20, 20]], manual=True, cleaning_label='manual')

        self.assertEqual(st['manual_wash_frames'], 0)
        self.assertEqual(st['effective_wash_frames'], 0)


if __name__ == '__main__':
    unittest.main()
