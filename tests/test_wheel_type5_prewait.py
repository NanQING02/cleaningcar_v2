import tempfile
import threading
import time
import types
import unittest

from cleaningcar.events import EventManager
from cleaningcar.wheel import WheelResultCache


class _DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx, vehicle_height=None):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class Type5WheelPrewaitTests(unittest.TestCase):
    def _manager(self, bind_wait=0.4):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        config = {
            "logic": {},
            "shadow_pool": {},
            "wheel": {"bind_wait_seconds": bind_wait},
            "event_capture_dir": self._tmp.name,
            "event_output_dir": self._tmp.name,
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(128, 128),
            zone_manager=_DummyZoneManager(),
            wheel_result_provider=WheelResultCache(bind_window_seconds=30.0, image_quality=80),
        )

    def _track(self, **extra):
        st = {
            'events': set(),
            'type2_qualified': True,
            'wheel_results_locked': {},
            'wheel_photo_history': {'left': {}, 'right': {}},
            'wheel_photo_seq': {'left': 0, 'right': 0},
        }
        st.update(extra)
        return st

    def test_should_prewait_requires_type2_qualified_open_track(self):
        mgr = self._manager()
        self.assertTrue(mgr._should_prewait_type5_wheel(self._track()))
        self.assertFalse(mgr._should_prewait_type5_wheel(self._track(closed=True)))
        self.assertFalse(mgr._should_prewait_type5_wheel(self._track(events={2, 5})))
        self.assertFalse(mgr._should_prewait_type5_wheel(self._track(type2_qualified=False)))

    def test_prewait_makes_type5_wait_nonblocking_after_grace(self):
        mgr = self._manager(bind_wait=0.4)
        st = self._track()

        mgr._ensure_type5_wheel_prewait(1, st, None)
        thread = st.get('wheel_prewait_thread')
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())
        self.assertGreater(float(st['wheel_bind_wait_deadline_ts']), time.time())

        # 等待预算耗尽（模拟4秒丢失宽限期已过）
        thread.join(timeout=1.5)
        time.sleep(0.05)

        started = time.time()
        mgr._wait_for_wheel_results_for_type5(1, st)
        self.assertLess(time.time() - started, 0.05)

    def test_wait_blocks_only_remaining_budget_when_thread_alive(self):
        mgr = self._manager(bind_wait=0.4)
        st = self._track()

        mgr._ensure_type5_wheel_prewait(1, st, None)
        started = time.time()
        mgr._wait_for_wheel_results_for_type5(1, st)
        elapsed = time.time() - started
        # 只等剩余预算（约0.4s），而不是从头再等一轮
        self.assertGreaterEqual(elapsed, 0.25)
        self.assertLess(elapsed, 1.0)

    def test_sync_fallback_without_prewait_keeps_wait_behavior(self):
        mgr = self._manager(bind_wait=0.3)
        st = self._track()

        started = time.time()
        mgr._wait_for_wheel_results_for_type5(1, st)
        self.assertGreaterEqual(time.time() - started, 0.2)

    def test_prewait_budget_resets_for_new_lost_episode(self):
        mgr = self._manager(bind_wait=0.4)
        st = self._track()

        mgr._ensure_type5_wheel_prewait(1, st, types.SimpleNamespace(lost_ts=100.0))
        first_deadline = float(st['wheel_bind_wait_deadline_ts'])
        thread = st['wheel_prewait_thread']
        thread.join(timeout=1.5)

        # 新的丢失周期：预算重置，线程重启
        mgr._ensure_type5_wheel_prewait(1, st, types.SimpleNamespace(lost_ts=200.0))
        new_deadline = float(st['wheel_bind_wait_deadline_ts'])
        self.assertGreater(new_deadline, first_deadline)
        self.assertIsNot(st['wheel_prewait_thread'], thread)

    def test_prewait_does_not_restart_within_same_episode(self):
        mgr = self._manager(bind_wait=0.4)
        st = self._track()
        lifecycle = types.SimpleNamespace(lost_ts=100.0)

        mgr._ensure_type5_wheel_prewait(1, st, lifecycle)
        first_thread = st['wheel_prewait_thread']
        mgr._ensure_type5_wheel_prewait(1, st, lifecycle)
        self.assertIs(st['wheel_prewait_thread'], first_thread)

    def test_flush_inactive_starts_prewait_for_lost_type2_track(self):
        mgr = self._manager(bind_wait=0.4)
        st = self._track(track_lost_grace=1)
        st['last_frame_idx'] = 0
        mgr.tracks[1] = st
        lifecycle = mgr.lifecycle_manager.create(1, 'car', capture_ts=1000.0)
        mgr.lifecycle_manager.touch(1, capture_ts=1000.0)
        mgr.lifecycle_manager.mark_lost(1, capture_ts=1000.0)

        mgr.flush_inactive(set(), 100, timeout_reason='track_lost', capture_ts=1000.05)

        self.assertIsNotNone(st.get('wheel_prewait_thread'))
        # 宽限期内轨迹未被移除
        self.assertIn(1, mgr.tracks)


if __name__ == '__main__':
    unittest.main()
