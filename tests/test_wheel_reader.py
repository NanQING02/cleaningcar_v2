import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cleaningcar.wheel import WheelDetectionService, WheelReaderThread, _LatestFrameSlot


class _StaticFrameCapture:
    def __init__(self, frame):
        self.frame = frame
        self.released = False

    @staticmethod
    def isOpened():
        return True

    def read(self):
        time.sleep(0.002)
        return True, self.frame.copy()

    def release(self):
        self.released = True


class WheelReaderThreadTests(unittest.TestCase):
    def test_wheel_reader_does_not_inherit_main_video_reader_target_fps(self):
        config = {
            "video": {"reader_target_fps": 20.0},
            "wheel": {"enabled": True},
        }

        service = WheelDetectionService(config, hw_decode=True)

        self.assertEqual(config["video"]["reader_target_fps"], 20.0)
        self.assertEqual(service.reader_args._config["video"]["reader_target_fps"], 0.0)

    def test_reconnects_when_successful_reads_keep_returning_same_frame(self):
        stop_event = threading.Event()
        frame_slot = _LatestFrameSlot()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        captures = []

        def fake_create_video_reader(source, reader_args):
            cap = _StaticFrameCapture(frame)
            captures.append(cap)
            if len(captures) >= 2:
                stop_event.set()
            return cap, {"decode_mode": "hw", "decode_backend": "fake"}

        reader = WheelReaderThread(
            side="left",
            source="rtsp://wheel-left",
            reader_args=SimpleNamespace(),
            frame_slot=frame_slot,
            stop_event=stop_event,
            reader_fail_threshold=5,
            reconnect_delay=0.2,
            stale_seconds=0.001,
            stale_check_interval_frames=1,
            stale_hash_size=8,
        )

        with patch("cleaningcar.wheel.create_video_reader", side_effect=fake_create_video_reader):
            reader.start()
            reader.join(timeout=2.0)

        stop_event.set()
        reader.join(timeout=1.0)

        self.assertFalse(reader.is_alive())
        self.assertGreaterEqual(len(captures), 2)
        self.assertTrue(captures[0].released)
        self.assertEqual(reader.last_reconnect_reason, "stale_frame")
        self.assertGreaterEqual(reader.reconnect_count, 1)

    def test_event_driven_reader_throttles_when_idle(self):
        stop_event = threading.Event()
        active_event = threading.Event()
        frame_slot = _LatestFrameSlot()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        captures = []

        class _FastCapture:
            def __init__(self):
                self.reads = 0
                self.released = False

            @staticmethod
            def isOpened():
                return True

            def read(self):
                self.reads += 1
                return True, frame.copy()

            def release(self):
                self.released = True

        def fake_create_video_reader(source, reader_args):
            cap = _FastCapture()
            captures.append(cap)
            return cap, {"decode_mode": "hw", "decode_backend": "fake"}

        reader = WheelReaderThread(
            side="left",
            source="rtsp://wheel-left",
            reader_args=SimpleNamespace(),
            frame_slot=frame_slot,
            stop_event=stop_event,
            reader_fail_threshold=5,
            reconnect_delay=0.2,
            stale_seconds=0.0,
            active_event=active_event,
            idle_fps=2.0,
        )

        with patch("cleaningcar.wheel.create_video_reader", side_effect=fake_create_video_reader):
            reader.start()
            deadline = time.time() + 1.0
            while reader.frames < 1 and time.time() < deadline:
                time.sleep(0.01)
            time.sleep(0.12)
            stop_event.set()
            reader.join(timeout=1.0)

        self.assertFalse(reader.is_alive())
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0].reads, 1)
        self.assertEqual(reader.frames, 1)
        self.assertEqual(reader.last_reader_mode, "idle")
        self.assertGreater(reader.idle_throttle_count, 0)


if __name__ == "__main__":
    unittest.main()
