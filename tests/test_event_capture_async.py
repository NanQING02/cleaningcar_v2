import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from cleaningcar.events import EventManager


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


class EventCaptureAsyncTests(unittest.TestCase):
    def _manager(self, capture_async):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        config = {
            "logic": {"event_capture_async": capture_async},
            "shadow_pool": {},
            "event_capture_dir": self._tmp.name,
            "event_output_dir": self._tmp.name,
        }
        return EventManager(
            config,
            fps=25.0,
            frame_size=(64, 64),
            zone_manager=_DummyZoneManager(),
        )

    def _frame(self):
        rng = np.random.default_rng(3)
        return rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def test_async_capture_returns_path_then_writes_file(self):
        mgr = self._manager(capture_async=True)
        frame = self._frame()

        path = mgr._save_event_capture(1, 7, 100, frame)

        self.assertTrue(path)
        self.assertTrue(mgr.wait_for_event_captures(timeout=5.0))
        self.assertTrue(Path(path).exists())
        self.assertGreater(Path(path).stat().st_size, 0)
        mgr.close()

    def test_async_capture_close_drains_queue(self):
        mgr = self._manager(capture_async=True)
        frame = self._frame()

        paths = [mgr._save_event_capture(1, 7, 100 + i, frame) for i in range(4)]

        mgr.close()
        for path in paths:
            self.assertTrue(Path(path).exists())

    def test_default_remains_synchronous(self):
        mgr = self._manager(capture_async=False)
        frame = self._frame()

        path = mgr._save_event_capture(2, 7, 200, frame)

        self.assertTrue(path)
        # 同步路径写盘即时完成
        self.assertTrue(Path(path).exists())
        self.assertIsNone(mgr._capture_worker)
        mgr.close()

    def test_async_capture_failure_on_none_frame(self):
        mgr = self._manager(capture_async=True)

        path = mgr._save_event_capture(1, 7, 100, None)

        self.assertEqual(path, '')
        mgr.close()


if __name__ == '__main__':
    unittest.main()
