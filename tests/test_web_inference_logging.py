import io
import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

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

    @classmethod
    def _manager_with_restart_guard(cls):
        manager = cls._manager_without_monitor_thread()
        manager.automatic_restart_times = deque(maxlen=64)
        manager.restart_history = deque(maxlen=32)
        manager.auto_restart_suspended = False
        manager.next_restart_at = None
        manager.last_restart_reason = ''
        manager.auto_restart_max_attempts = 3
        manager.auto_restart_window_seconds = 600.0
        manager.auto_restart_backoff_seconds = 5.0
        manager.auto_restart_backoff_max_seconds = 60.0
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

    def test_reader_reconnect_status_is_degraded_instead_of_progress_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_path = Path(tmpdir) / 'heartbeat.json'
            heartbeat_path.write_text(json.dumps({
                'timestamp': 200.0,
                'last_progress_ts': 50.0,
                'pid': 123,
                'launch_id': 'launch-a',
                'status': 'reader_reconnect',
            }), encoding='utf-8')
            manager = InferenceManager.__new__(InferenceManager)
            manager.heartbeat_path = heartbeat_path
            manager.last_start = 1.0
            manager.runtime_namespace_key = 'cam-a'
            manager.launch_id = 'launch-a'
            manager.process = SimpleNamespace(pid=123, poll=lambda: None)
            manager.heartbeat_timeout_seconds = 30.0
            manager.progress_timeout_seconds = 90.0
            manager.heartbeat_startup_grace_seconds = 90.0

            result = manager._check_heartbeat_locked(now=200.0)

            self.assertTrue(result['healthy'])
            self.assertTrue(result['degraded'])
            self.assertEqual(result['reason'], 'reader_reconnect')

    def test_running_status_still_detects_progress_stall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_path = Path(tmpdir) / 'heartbeat.json'
            heartbeat_path.write_text(json.dumps({
                'timestamp': 200.0,
                'last_progress_ts': 50.0,
                'pid': 123,
                'launch_id': 'launch-a',
                'status': 'running',
            }), encoding='utf-8')
            manager = InferenceManager.__new__(InferenceManager)
            manager.heartbeat_path = heartbeat_path
            manager.last_start = 1.0
            manager.runtime_namespace_key = 'cam-a'
            manager.launch_id = 'launch-a'
            manager.process = SimpleNamespace(pid=123, poll=lambda: None)
            manager.heartbeat_timeout_seconds = 30.0
            manager.progress_timeout_seconds = 90.0
            manager.heartbeat_startup_grace_seconds = 90.0

            result = manager._check_heartbeat_locked(now=200.0)

            self.assertFalse(result['healthy'])
            self.assertEqual(result['reason'], 'progress_stale')

    def test_auto_restart_uses_exponential_backoff_and_circuit_breaker(self):
        manager = self._manager_with_restart_guard()

        self.assertTrue(manager._schedule_auto_restart_locked('exit:1', now=100.0))
        self.assertEqual(manager.next_restart_at, 105.0)
        self.assertTrue(manager._schedule_auto_restart_locked('exit:1', now=110.0))
        self.assertEqual(manager.next_restart_at, 120.0)
        self.assertTrue(manager._schedule_auto_restart_locked('exit:1', now=130.0))
        self.assertEqual(manager.next_restart_at, 150.0)
        self.assertFalse(manager._schedule_auto_restart_locked('exit:1', now=160.0))
        self.assertTrue(manager.auto_restart_suspended)
        self.assertIsNone(manager.next_restart_at)
        self.assertEqual(manager.restart_history[-1]['action'], 'suspended')

    def test_manual_restart_guard_reset_clears_suspension_and_history(self):
        manager = self._manager_with_restart_guard()
        manager.automatic_restart_times.extend((100.0, 110.0, 120.0))
        manager.restart_history.append({'action': 'suspended'})
        manager.auto_restart_suspended = True
        manager.next_restart_at = 130.0
        manager.last_restart_reason = 'heartbeat:stale'

        manager._reset_restart_guard_locked()

        self.assertEqual(list(manager.automatic_restart_times), [])
        self.assertEqual(list(manager.restart_history), [])
        self.assertFalse(manager.auto_restart_suspended)
        self.assertIsNone(manager.next_restart_at)
        self.assertEqual(manager.last_restart_reason, '')

    def test_process_exit_reason_preserves_terminal_reader_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            heartbeat_path = Path(tmpdir) / 'heartbeat.json'
            heartbeat_path.write_text(json.dumps({
                'status': 'reader_failed',
                'failure_reason': 'main reader reconnect exhausted after 20 attempts',
            }), encoding='utf-8')
            manager = self._manager_with_restart_guard()
            manager.heartbeat_path = heartbeat_path

            reason = manager._process_exit_reason_locked(1)

            self.assertEqual(
                reason,
                'reader_failed:main reader reconnect exhausted after 20 attempts',
            )


if __name__ == "__main__":
    unittest.main()
