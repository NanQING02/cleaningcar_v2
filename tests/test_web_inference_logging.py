import io
import threading
import unittest
from collections import deque
from pathlib import Path

from web.inference import (
    InferenceManager,
    _build_inference_log_name,
    _inference_routine_category,
)


class InferenceLoggingTests(unittest.TestCase):
    @staticmethod
    def _manager_without_monitor_thread():
        manager = InferenceManager.__new__(InferenceManager)
        manager.log_buffer = deque(maxlen=800)
        manager.log_lock = threading.Lock()
        manager._routine_last_emit = {}
        manager._routine_suppressed = {}
        return manager

    def test_log_name_separates_config_device_and_launch(self):
        started_at = 1_789_632_000.0

        first = _build_inference_log_name(
            Path("configs/config1.json"),
            "RK3588-DEV1",
            "abc123",
            started_at,
        )
        second = _build_inference_log_name(
            Path("configs/config2.json"),
            "RK3588-DEV2",
            "def456",
            started_at,
        )

        self.assertNotEqual(first, second)
        self.assertIn("config1", first)
        self.assertIn("RK3588-DEV1", first)
        self.assertIn("abc123", first)

    def test_routine_classifier_keeps_events_and_errors_unthrottled(self):
        self.assertEqual(_inference_routine_category("[perf] fps=20"), "perf")
        self.assertEqual(_inference_routine_category("[diag] flags=ok"), "diag")
        self.assertEqual(_inference_routine_category("[monitor] risk=ok"), "monitor")
        self.assertEqual(
            _inference_routine_category("W Query dynamic range failed. Ret code: -1"),
            "rknn_static",
        )
        self.assertEqual(_inference_routine_category("[EVENT] track=1 type=5"), "")
        self.assertEqual(_inference_routine_category("Traceback (most recent call last):"), "")
        self.assertEqual(_inference_routine_category("[reader] reconnect attempt #2"), "")

    def test_routine_lines_are_throttled_but_summary_and_important_lines_remain(self):
        manager = self._manager_without_monitor_thread()
        output = io.StringIO()

        self.assertTrue(manager._write_captured_line("[perf] fps=20", output, "config1", now=0.0))
        self.assertFalse(manager._write_captured_line("[perf] fps=21", output, "config1", now=1.0))
        self.assertTrue(manager._write_captured_line("[EVENT] type=5", output, "config1", now=2.0))
        self.assertTrue(manager._write_captured_line("[perf] fps=22", output, "config1", now=61.0))

        text = output.getvalue()
        self.assertIn("[perf] fps=20", text)
        self.assertNotIn("[perf] fps=21", text)
        self.assertIn("[EVENT] type=5", text)
        self.assertIn("已省略 1 条", text)
        self.assertIn("[perf] fps=22", text)


if __name__ == "__main__":
    unittest.main()
