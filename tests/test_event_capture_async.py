import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cleaningcar.events import EventManager, EventUploader


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

    def test_path_payload_is_preserved_while_async_file_is_pending(self):
        mgr = self._manager(capture_async=True)
        started = threading.Event()
        release = threading.Event()
        original = mgr._encode_event_capture

        def delayed_encode(**kwargs):
            started.set()
            release.wait(timeout=2.0)
            return original(**kwargs)

        mgr._encode_event_capture = delayed_encode
        path = mgr._save_event_capture(1, 7, 300, self._frame())
        self.assertTrue(started.wait(timeout=1.0))
        self.assertFalse(Path(path).exists())
        self.assertEqual(mgr._prepare_capture_image(path), path)
        self.assertFalse(mgr.wait_for_event_captures(timeout=0.05))
        release.set()
        self.assertTrue(mgr.wait_for_event_captures(timeout=2.0))
        mgr.close()

    def test_event_uploader_retries_until_capture_path_exists(self):
        uploader = EventUploader()
        uploader.url = 'http://127.0.0.1/event'
        with tempfile.TemporaryDirectory() as tmpdir:
            capture = Path(tmpdir) / 'pending.jpg'
            with self.assertRaises(FileNotFoundError):
                uploader._send({'captureImage': str(capture)})

            capture.write_bytes(b'jpeg')
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = b'ok'
            with patch('cleaningcar.events.urllib.request.urlopen', return_value=response) as mocked:
                uploader._send({'captureImage': str(capture)})
            mocked.assert_called_once()


if __name__ == '__main__':
    unittest.main()
