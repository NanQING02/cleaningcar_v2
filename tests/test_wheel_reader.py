import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from cleaningcar.wheel import WheelReaderThread, _LatestFrameSlot


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


if __name__ == "__main__":
    unittest.main()
